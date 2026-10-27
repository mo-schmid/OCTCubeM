
import os
from collections.abc import Callable
from pathlib import Path

import lightning.pytorch as pl
import monai.transforms as mt
import pandas as pd
from monai.data import Dataset, PersistentDataset
from monai.data.utils import pickle_hashing
from torch.utils.data import DataLoader, default_collate

from retimod.data.splitting import grouped_stratified_split
from retimod.data.transforms.defaults import normalize


class IndexFileDataModule(pl.LightningDataModule):
    """Base class for datamodules that have a index file beside the data.

    Uses MONAI Dataset internally.
    Generated dataloaders return a dict with whatever columns `_process_index_file`
    produced (always `image`, `label`, `group`)

    Args:
        view: what one sample is. `volume` yields one sample per eye carrying
            `num_slices` B-scans; `bscan` yields one sample per B-scan, for a
            true 2D protocol. Images stay `(C, D, H, W)` either way -- a single
            B-scan is the `D == 1` case.
        train_augment: Applied to training samples after `_build_preprocess`, on uint8
            images; `None` trains on the preprocessed samples as they are.
        eval_augment: The same for val/test/full samples.
        normalize_on_device: Ship uint8 batches and scale them to `[0, 1]` in
            `on_after_batch_transfer` instead of in the workers -- 4x less data to move. Only
            a Trainer calls that hook; a hand-written loop has to call it itself.
    """

    CACHE_NAME: str
    VIEWS = ("volume", "bscan")
    TRAIN_FRACTION = 0.6
    VAL_FRACTION = 0.2
    TEST_FRACTION = 0.2

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        seed: int = 42,
        cache_dir: str | None = "auto",
        view: str = "volume",
        train_augment: Callable | None = None,
        eval_augment: Callable | None = None,
        normalize_on_device: bool = False,
        **kwargs,
    ):
        super().__init__()
        if view not in self.VIEWS:
            raise ValueError(f"Unknown view '{view}'. Available: {', '.join(self.VIEWS)}")
        # the transforms are objects, not values: pickling them into hparams is useless
        self.save_hyperparameters(ignore=["train_augment", "eval_augment"])
        self.train_augment = train_augment
        self.eval_augment = eval_augment
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None

        self._train_transform = None
        self._eval_transform = None

    @property
    def sample_depth(self) -> int:
        """How many B-scans one sample carries, for subclasses to size their
        depth pad/crop with. Requires subclasses to declare `num_slices`.
        """
        return 1 if self.hparams.view == "bscan" else self.hparams.num_slices

    @property
    def label_names(self) -> list[str] | None:
        """What each class id of `label` means, from the `_label_to_idx` a subclass's
        `_process_index_file` builds; `None` before `setup` or without one."""
        label_to_idx = getattr(self, "_label_to_idx", None)
        if label_to_idx is None:
            return None
        names = sorted(label_to_idx, key=label_to_idx.get)
        # some datasets label with numbers (e.g. OIMHS's stage)
        return [str(name) for name in names]

    def _read_index_file(self, data_dir: Path):
        index_path = data_dir / "index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"Could not find dataset index at {index_path}")

        df = pd.read_csv(index_path)
        return df

    def _process_index_file(self, df_index: pd.DataFrame) -> pd.DataFrame:
        return df_index

    def _create_data_splits(self, df_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if "label" not in df_index:
            raise ValueError("Failed to create data splits. Missing 'label' column in dataframe.")

        if "group" not in df_index:
            raise ValueError("Failed to create data splits. Missing 'group' column in dataframe.")

        train_idx, val_idx, test_idx = grouped_stratified_split(
            df_index["label"].to_numpy(),
            df_index["group"].to_numpy(),
            fractions=[self.TRAIN_FRACTION, self.VAL_FRACTION, self.TEST_FRACTION],
            random_state=self.hparams.seed,
        )
        df_train = df_index.iloc[train_idx].reset_index(drop=True)
        df_val = df_index.iloc[val_idx].reset_index(drop=True)
        df_test = df_index.iloc[test_idx].reset_index(drop=True)
        return df_train, df_val, df_test

    def _build_preprocess(self) -> list:
        """The deterministic per-sample transforms: the loader plus `transforms.defaults.standardize`."""
        raise NotImplementedError

    def _compose(self, preprocess: list, augment: Callable | None) -> mt.Compose:
        if augment is None:
            return mt.Compose(preprocess)
        return mt.Compose([*preprocess, augment])

    def _build_dataset(self, records: pd.DataFrame, transform: mt.Compose):
        # `PersistentDataset` hashes and caches only the deterministic prefix up to the
        # first random transform, so train and eval share one uint8 cache
        data = records.to_dict("records")
        cache_dir = self._resolve_cache_dir()
        if cache_dir is not None:
            return PersistentDataset(
                data=data,
                transform=transform,
                cache_dir=cache_dir,
                hash_transform=pickle_hashing,
            )
        return Dataset(data=data, transform=transform)

    def _resolve_cache_dir(self):
        cache_dir = self.hparams.cache_dir
        if cache_dir != "auto":
            return cache_dir
        # SLURM sets TMPDIR to node-local scratch for the duration of the job
        # (see TmpFS in slurm.conf); outside SLURM, TMPDIR is unset and we fall
        # back to no caching rather than guessing at a local path.
        tmpdir = os.environ.get("TMPDIR")
        return str(Path(tmpdir) / f"retimod_{self.CACHE_NAME}_cache") if tmpdir else None

    def _build_dataloader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            collate_fn=self._collate,
            pin_memory=True,
        )

    def _collate(self, samples: list) -> dict:
        batch = default_collate(samples)
        if self.hparams.normalize_on_device:
            return batch
        return normalize(batch, self._image_keys())

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        if not self.hparams.normalize_on_device:
            return batch
        return normalize(batch, self._image_keys())

    def _image_keys(self) -> list[str]:
        """`image`, plus whatever image keys (e.g. multi-crop views) the train augment emits."""
        extra = getattr(self.train_augment, "image_keys", [])
        return ["image", *extra]

    def setup(self, stage=None):
        df_index = self._read_index_file(Path(self.hparams.data_dir))
        df_index = self._process_index_file(df_index)
        df_train, df_val, df_test = self._create_data_splits(df_index)

        preprocess = self._build_preprocess()
        self._train_transform = self._compose(preprocess, self.train_augment)
        self._eval_transform = self._compose(preprocess, self.eval_augment)

        if stage is None or stage == "fit":
            self.train_dataset = self._build_dataset(df_train, self._train_transform)

        if stage is None or stage in ["fit", "validate"]:
            self.val_dataset = self._build_dataset(df_val, self._eval_transform)

        if stage is None or stage == "test":
            self.test_dataset = self._build_dataset(df_test, self._eval_transform)

        if stage is None or stage == "predict":
            self.full_dataset = self._build_dataset(df_index, self._eval_transform)

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._build_dataloader(self.test_dataset, shuffle=False)

    def full_dataloader(self):
        return self._build_dataloader(self.full_dataset, shuffle=False)


def encode_ids(keys: pd.Series) -> pd.Series:
    """Maps string identifiers to integer codes, so a batch collates them into a tensor.

    Codes follow the sorted order of the keys, so a splitter that sorts its
    groups (as `StratifiedGroupKFold` does) draws the same folds from either.
    """
    codes, _ = pd.factorize(keys, sort=True)
    return pd.Series(codes, index=keys.index)
