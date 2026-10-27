import io
from pathlib import Path

import numpy as np
import pandas as pd

from octcubem.datamodules.webdataset_base import WebDatasetDataModule
from octcubem.transforms.defaults import InPlaneAugment, standardize

DATA_DIR = "data/alleye-macula_n100"


class AlleyeMaculaDataModule(WebDatasetDataModule):
    """Streaming MAE-pretraining datamodule for the alleye-macula dataset.

    A 100-sample, development-scale stand-in for the full ~300k-volume dataset,
    sharing the same webdataset shard layout: one `<hash>.npy` macula OCT volume
    (uint8, `(D, H, W)`) per sample, packed into per-split tar shards alongside a
    `<hash>.json` DICOM metadata sidecar (unused here -- there's no pretraining label).
    """

    def __init__(
        self,
        *args,
        data_dir=DATA_DIR,
        image_size: int | tuple[int, int] = 224,
        num_slices: int = 49,
        train_folds=(0,1,2,3,4,5,6,7),
        val_folds=(8,),
        test_folds=(9,),
        resize_mode: str = "area",
        **kwargs,
    ):
        super().__init__(
            *args,
            data_dir=data_dir,
            image_size=image_size,
            num_slices=num_slices,
            train_folds=train_folds,
            val_folds=val_folds,
            test_folds=test_folds,
            resize_mode=resize_mode,
            **kwargs,
        )

    def _list_shards(self, data_dir: Path) -> pd.DataFrame:
        overview_path = data_dir / "overview.csv"
        if not overview_path.exists():
            raise FileNotFoundError(f"Could not find dataset index at {overview_path}")

        df = pd.read_csv(overview_path)
        if "split" not in df.columns:
            raise ValueError(f"{overview_path} has no `split` column -- dataset provides no fold to split on")

        # `split` names the patient-exclusive `split-XXX` folder a shard lives in --
        # constant per shard_path, since a shard never spans two folders
        shards = df.groupby("shard_path").agg(num_samples=("sample_id", "count"), fold=("split", "first")).reset_index()
        shards["shard_path"] = shards["shard_path"].map(lambda p: str(data_dir / p))
        return shards

    def _decode(self, sample: dict) -> dict:
        return {"image": np.load(io.BytesIO(sample["npy"])), "sample_id": sample["__key__"]}

    def _build_preprocess(self) -> list:
        # the decoded array has no metadata to say it lacks a channel axis
        return standardize(
            ["image"],
            self.hparams.num_slices,
            self.hparams.image_size,
            channel_dim="no_channel",
            resize_mode=self.hparams.resize_mode,
        )

    def _default_train_augment(self) -> InPlaneAugment:
        image_size = self.hparams.image_size
        height = image_size if isinstance(image_size, int) else image_size[0]
        return InPlaneAugment(max_translation=height * 0.1)
