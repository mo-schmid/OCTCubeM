import math
from collections.abc import Callable, Sequence
from pathlib import Path

import lightning.pytorch as pl
import monai.transforms as mt
import pandas as pd
import torch
import webdataset as wds
from monai.data import list_data_collate

from octcubem.transforms.defaults import normalize


class WebDatasetDataModule(pl.LightningDataModule):
    """Base class for streaming, webdataset-format pretraining datasets, decoded and
    transformed with MONAI.

    Unlike `IndexFileDataModule`, splits are assigned whole folds at a time rather than
    individual shards or rows: the dataset provides its own patient-exclusive fold
    grouping (a shard is not necessarily patient-exclusive itself, but never spans two
    folds), so assigning a fold as a unit is group-safe by construction and avoids
    random-access reads into tar files.

    Training reads an infinite, node/worker-sharded, resampled stream, so DDP ranks
    never stall on an uneven shard count; val/test iterate their shard set exactly
    once (not resampled) so metrics stay comparable across checkpoints/runs. Every rank
    reads the full val/test set.

    Under DDP, set `Trainer(use_distributed_sampler=False)`: shard distribution across
    ranks is handled by `nodesplitter` in `_build_dataloader`, not by Lightning's
    default `DistributedSampler`, which doesn't apply to a streaming `IterableDataset`.

    Args:
        train_augment: Applied to training samples after `_build_preprocess`, on uint8
            images. `None` falls back to `_default_train_augment`.
        eval_augment: The same for val/test samples; `None` applies nothing.
        normalize_on_device: Ship uint8 batches and scale them to `[0, 1]` in
            `on_after_batch_transfer` instead of in the workers -- 4x less data to move. Only
            a Trainer calls that hook; a hand-written loop has to call it itself.
        device_augment: Applied to each training batch, in `on_after_batch_transfer`, once
            it is on the device -- e.g. `BatchedMultiView3D`, which needs the uint8 `image`
            and so `normalize_on_device`. Setting it drops the default `train_augment`;
            pass one explicitly to run both. Outside a Trainer every batch counts as a
            training batch.
    """

    def __init__(
        self,
        data_dir: str,
        train_folds: Sequence,
        val_folds: Sequence,
        test_folds: Sequence,
        batch_size: int = 32,
        num_workers: int = 8,
        image_size: int | tuple[int, int] = 224,
        num_slices: int = 49,
        steps_per_epoch: int | None = None,
        shard_shuffle_buffer: int = 32,
        train_augment: Callable | None = None,
        eval_augment: Callable | None = None,
        normalize_on_device: bool = False,
        device_augment: Callable | None = None,
        **kwargs,
    ):
        super().__init__()
        if device_augment is not None and not normalize_on_device:
            raise ValueError("`device_augment` works on the uint8 batch, so it needs `normalize_on_device=True`")

        # the transforms are objects, not values: pickling them into hparams is useless
        self.save_hyperparameters(ignore=["train_augment", "eval_augment", "device_augment"])
        self.train_augment = train_augment
        self.eval_augment = eval_augment
        self.device_augment = device_augment
        self._all_shards = None
        self._train_shards = None
        self._val_shards = None
        self._test_shards = None
        self._train_len = None
        self._train_transform = None
        self._eval_transform = None

    def setup(self, stage=None):
        self._all_shards = self._list_shards(Path(self.hparams.data_dir))
        self._train_shards, self._val_shards, self._test_shards = self._split_shards(self._all_shards)
        self._train_len = int(self._train_shards["num_samples"].sum())

        if self.train_augment is None and self.device_augment is None:
            self.train_augment = self._default_train_augment()
        preprocess = self._build_preprocess()
        self._train_transform = self._compose(preprocess, self.train_augment)
        self._eval_transform = self._compose(preprocess, self.eval_augment)

    def train_dataloader(self):
        return self._build_dataloader(self._train_shards, self._train_transform, resampled=True)

    def val_dataloader(self):
        return self._build_dataloader(self._val_shards, self._eval_transform, resampled=False)

    def test_dataloader(self):
        return self._build_dataloader(self._test_shards, self._eval_transform, resampled=False)

    def full_dataloader(self):
        """Every shard, whatever its fold, read once -- what a frozen encoder's feature extraction iterates.

        With several workers the samples arrive interleaved across shards, so a
        consumer that needs a stable order must emit its own sample index.
        """
        return self._build_dataloader(self._all_shards, self._eval_transform, resampled=False)

    @property
    def sample_depth(self) -> int:
        """How many B-scans one sample carries."""
        return self.hparams.num_slices

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        # the device augment's views come out in [0, 1] already, and it has consumed
        # `image`; `normalize` skips missing keys, so it only scales what is left
        training = self.trainer is None or self.trainer.training
        if self.device_augment is not None and training:
            batch = self.device_augment(batch)
        if not self.hparams.normalize_on_device:
            return batch
        return normalize(batch, self._image_keys())

    def _compose(self, preprocess: list, augment: Callable | None) -> mt.Compose:
        if augment is None:
            return mt.Compose(preprocess)
        return mt.Compose([*preprocess, augment])

    def _build_dataloader(self, shards: pd.DataFrame, transform: mt.Compose, resampled: bool):
        # shard order is randomized by `resampled` itself for train, and left
        # deterministic for val/test -- `shardshuffle` would be redundant either way.
        # val/test unsplit, so ranks run equal batch counts; `None` as the default raises under DDP
        nodesplitter = wds.split_by_node if resampled else None
        pipeline = wds.WebDataset(
            shards["shard_path"].tolist(),
            resampled=resampled,
            shardshuffle=False,
            nodesplitter=nodesplitter,
        )
        if resampled:
            pipeline = pipeline.shuffle(self.hparams.shard_shuffle_buffer)
        pipeline = pipeline.map(self._decode).map(transform)

        # a read-once shard set is split across workers, and a worker left without a
        # shard makes webdataset raise -- so never start more workers than shards
        num_workers = self.hparams.num_workers
        if not resampled:
            num_workers = min(num_workers, len(shards))

        loader = wds.WebLoader(
            pipeline,
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
            collate_fn=self._collate,
            worker_init_fn=self._make_worker_init(transform),
            pin_memory=True,
        )
        # a resampled stream has no natural end, so it needs an explicit epoch length;
        # val/test stop on their own once their (non-resampled) shard set is exhausted
        if not resampled:
            return loader

        steps = self._steps_per_epoch()
        loader = loader.with_epoch(steps)
        # `with_epoch` bounds the stream but leaves the loader without `__len__`,
        # which Lightning needs to size a per-step LR schedule (without it,
        # `Trainer.estimated_stepping_batches` is infinite). `with_length` is
        # webdataset's own workaround for exactly that; it reports the length
        # `with_epoch` already enforces, so the two never disagree.
        return loader.with_length(steps, silent=True)

    def _collate(self, samples: list) -> dict:
        batch = list_data_collate(samples)
        if self.hparams.normalize_on_device:
            return batch
        return normalize(batch, self._image_keys())

    def _image_keys(self) -> list[str]:
        """`image`, plus whatever image keys (e.g. multi-crop views) the train augment emits."""
        extra = getattr(self.train_augment, "image_keys", [])
        return ["image", *extra]

    def _steps_per_epoch(self) -> int:
        """Optimizer steps one rank takes per epoch.

        The count is per rank, not global: every rank draws its own `with_epoch`
        stream, so dividing by `world_size` is what keeps one epoch equal to one
        pass over the training set however many GPUs are used -- and with it what
        `trainer.max_epochs` buys, including the LR schedule's horizon.
        """
        if self.hparams.steps_per_epoch is not None:
            return self.hparams.steps_per_epoch

        world_size = self.trainer.world_size if self.trainer is not None else 1
        samples_per_rank = self._train_len / world_size
        return max(1, math.ceil(samples_per_rank / self.hparams.batch_size))

    @staticmethod
    def _make_worker_init(transform: mt.Compose):
        # MONAI's random transforms carry their own RNG state, which a forked
        # dataloader worker inherits verbatim from the parent -- without reseeding
        # per worker, every worker would draw identical augmentations.
        def _init(worker_id):
            info = torch.utils.data.get_worker_info()
            transform.set_random_state(seed=info.seed % (2**32))

        return _init

    def _split_shards(self, df_shards: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Assigns each shard's fold to train/val/test per `hparams.*_folds`."""
        if "fold" not in df_shards.columns:
            raise ValueError(
                "Dataset provides no fold information to split on -- `_list_shards` must "
                "return a `fold` column identifying each shard's patient-exclusive group."
            )

        train_shards = df_shards[df_shards["fold"].isin(self.hparams.train_folds)]
        val_shards = df_shards[df_shards["fold"].isin(self.hparams.val_folds)]
        test_shards = df_shards[df_shards["fold"].isin(self.hparams.test_folds)]
        return train_shards, val_shards, test_shards

    def _list_shards(self, data_dir: Path) -> pd.DataFrame:
        """Returns one row per shard: `shard_path` (path `WebDataset` can open),
        `num_samples`, and `fold` -- the dataset's own patient-exclusive grouping id.
        """
        raise NotImplementedError

    def _decode(self, sample: dict) -> dict:
        """Decodes one raw webdataset sample (extension -> bytes) into the dict a
        MONAI dictionary transform expects, e.g. `{"image": np.ndarray}`.
        """
        raise NotImplementedError

    def _build_preprocess(self) -> list:
        """The deterministic per-sample transforms after `_decode`, typically `transforms.defaults.standardize`."""
        raise NotImplementedError

    def _default_train_augment(self) -> Callable | None:
        """The train augment used when none is passed in; `None` applies nothing."""
        return None
