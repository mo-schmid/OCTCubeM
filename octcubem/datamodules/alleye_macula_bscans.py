import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import webdataset as wds

from octcubem.datamodules.alleye_macula import DATA_DIR, AlleyeMaculaDataModule
from octcubem.datamodules.self_paced import HardnessTable
from octcubem.transforms.defaults import standardize


class AlleyeMaculaBscanDataModule(AlleyeMaculaDataModule):
    """The alleye-macula volumes as a stream of single B-scans, for OCTCube's 2D pretraining branch.

    Reads the same shards and folds as `AlleyeMaculaDataModule`, but splits every
    volume into its B-scans. Samples are `(1, 1, H, W)` uint8 `image`s, i.e.
    single-B-scan volumes, together with the volume's `sample_id` (its webdataset key,
    the `sample_id` column of overview.csv) and the B-scan's `slice` index in the
    stored volume -- what self-paced learning needs to look up a B-scan's hardness.

    Training yields `bscans_per_volume` random B-scans of every volume (all of them by
    default), mixed across volumes by a `bscan_shuffle_buffer`-deep shuffle, since
    neighbouring B-scans of one volume are nearly identical. Val/test yield one fixed
    B-scan per volume, picked by hashing its key, so the set is the same in every run.

    Like the original's 2D transform, there is no train augmentation by default.

    With `self_paced`, training only yields the B-scans selected in `hardness`, a
    `HardnessTable` over the train volumes that starts from a random
    `spl_init_fraction` of them. Whoever scores the B-scans updates the table and its
    selection between epochs; a train loader built afterwards (e.g. with
    `Trainer(reload_dataloaders_every_n_epochs=1)`) uses the new selection. The table is
    part of the datamodule's `state_dict`, so a resumed run keeps it.

    Args:
        bscans_per_volume: B-scans drawn per volume and visit in training; `None` takes all.
        bscan_shuffle_buffer: B-scans held per worker for shuffling, each about
            `H * W` bytes before the resize.
        self_paced: Restrict training to the B-scans selected in `hardness`.
        spl_init_fraction: The fraction of B-scans selected, at random, before any is scored.
        spl_seed: Seed of that random selection, the same in every process.
    """

    def __init__(
        self,
        *args,
        data_dir=DATA_DIR,
        image_size: int | tuple[int, int] = 512,
        bscans_per_volume: int | None = None,
        bscan_shuffle_buffer: int = 500,
        resize_mode: str = "bicubic",
        self_paced: bool = False,
        spl_init_fraction: float = 0.2,
        spl_seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            *args,
            data_dir=data_dir,
            image_size=image_size,
            num_slices=1,
            bscans_per_volume=bscans_per_volume,
            bscan_shuffle_buffer=bscan_shuffle_buffer,
            resize_mode=resize_mode,
            self_paced=self_paced,
            spl_init_fraction=spl_init_fraction,
            spl_seed=spl_seed,
            **kwargs,
        )
        self.hardness = None

    def setup(self, stage=None):
        super().setup(stage)
        # kept across `setup` calls, and replaced by `load_state_dict` on a resume
        if self.hparams.self_paced and self.hardness is None:
            df = pd.read_csv(Path(self.hparams.data_dir) / "overview.csv", usecols=["sample_id", "split", "frames"])
            df = df[df["split"].isin(self.hparams.train_folds)]
            self.hardness = HardnessTable(df["sample_id"].tolist(), df["frames"].tolist())
            self.hardness.select_random(self.hparams.spl_init_fraction, seed=self.hparams.spl_seed)

    @property
    def num_train_bscans(self) -> int:
        """B-scans the train stream draws from: the selected ones, under self-paced learning."""
        if self.hardness is not None:
            return self.hardness.num_selected
        return self._train_len

    def state_dict(self) -> dict:
        if self.hardness is None:
            return {}
        return {"hardness": self.hardness.state_dict()}

    def load_state_dict(self, state_dict: dict):
        if "hardness" in state_dict:
            self.hardness.load_state_dict(state_dict["hardness"])

    def _list_shards(self, data_dir: Path) -> pd.DataFrame:
        # `num_samples` counts what a shard yields per pass in training -- B-scans, not
        # volumes -- since `_steps_per_epoch` sizes an epoch from it
        shards = super()._list_shards(data_dir)
        df = pd.read_csv(data_dir / "overview.csv", usecols=["shard_path", "frames"])
        per_volume = df["frames"]
        if self.hparams.bscans_per_volume is not None:
            per_volume = per_volume.clip(upper=self.hparams.bscans_per_volume)
        bscans = per_volume.groupby(df["shard_path"]).sum()
        shards["num_samples"] = shards["shard_path"].map(lambda p: bscans[str(Path(p).relative_to(data_dir))])
        return shards

    def _expand(self, pipeline: wds.WebDataset, training: bool) -> wds.WebDataset:
        if not training:
            return pipeline.compose(_fixed_bscan)
        selection = None
        if self.hardness is not None:
            # a snapshot: the table's selection changes between epochs, a built loader's must not
            selection = (self.hardness.rows, self.hardness.selection.copy())
        pipeline = pipeline.compose(_split_into_bscans(self.hparams.bscans_per_volume, selection))
        return pipeline.shuffle(self.hparams.bscan_shuffle_buffer)

    def _build_preprocess(self) -> list:
        # a B-scan is kept as a one-slice `(1, H, W)` volume, so it has no channel axis
        return standardize(
            ["image"],
            self.sample_depth,
            self.hparams.image_size,
            channel_dim="no_channel",
            resize_mode=self.hparams.resize_mode,
        )

    def _default_train_augment(self) -> None:
        return None


def _split_into_bscans(bscans_per_volume: int | None, selection: tuple[dict, np.ndarray] | None = None):
    """A `compose` stage yielding every B-scan of each volume, or `bscans_per_volume` random ones.

    `selection`, a `(rows, mask)` pair of a `HardnessTable`, restricts both to the
    selected B-scans; a volume without any is skipped.
    """

    def _split(samples):
        rng = np.random.default_rng()
        for sample in samples:
            slices = np.arange(len(sample["image"]))
            if selection is not None:
                rows, mask = selection
                slices = slices[mask[rows[sample["sample_id"]], : len(slices)]]
            if bscans_per_volume is not None and bscans_per_volume < len(slices):
                slices = np.sort(rng.choice(slices, bscans_per_volume, replace=False))
            for index in slices:
                yield _bscan(sample, int(index))

    return _split


def _fixed_bscan(samples):
    """A `compose` stage yielding one B-scan per volume, the same one in every run."""
    for sample in samples:
        yield _bscan(sample, zlib.crc32(sample["sample_id"].encode()) % len(sample["image"]))


def _bscan(sample: dict, index: int) -> dict:
    """B-scan `index` of a decoded volume sample, as a one-slice volume.

    It gets a key of its own, since webdataset's `map` carries `__key__` over to the
    transformed sample and the collate expects one.
    """
    return {
        # a copy, so a held B-scan doesn't keep its whole volume alive in the shuffle buffer
        "image": sample["image"][index : index + 1].copy(),
        "sample_id": sample["sample_id"],
        "slice": index,
        "__key__": "{}/{}".format(sample["sample_id"], index),
    }
