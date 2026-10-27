import math
from pathlib import Path

import monai.transforms as mt
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler

from octcubem.datamodules.base import IndexFileDataModule
from octcubem.datamodules.readers import ImageSeriesReader
from octcubem.transforms.defaults import standardize

DATA_DIR = "Pre-training/ucsd_kermany_2018"


class Kermany2018DataModule(IndexFileDataModule):
    """2D pretraining datamodule for the UCSD Kermany 2018 OCT dataset (CNV/DME/DRUSEN/NORMAL).

    The dataset is described by an index.csv file (`subject_id, patient_id, diagnosis,
    split, image_nr, width, height, image_path`) and a folder of JPEG B-scans, one row
    per B-scan. Samples are `(1, 1, H, W)` uint8 `image`s, i.e. single-B-scan volumes.

    The dataset's own train/test split is kept rather than re-split: train is what
    OCTCube's joint pretraining used (`ImageFolder(<dir>/train)`), and the 1,000 test
    B-scans serve as a small val set. There is no test set.

    Training draws an infinite with-replacement stream cut into epochs of
    `steps_per_epoch` batches, so it can run in lockstep with the 3D stream.
    `steps_per_epoch` is per rank, as in `WebDatasetDataModule`; `None` is one pass
    over the train set across all ranks.
    """

    CACHE_NAME = "ucsd_kermany_2018"
    VIEWS = ("bscan",)

    def __init__(
        self,
        *args,
        data_dir=DATA_DIR,
        image_size: int | tuple[int, int] = 512,
        steps_per_epoch: int | None = None,
        resize_mode: str = "trilinear",
        **kwargs,
    ):
        super().__init__(
            *args,
            data_dir=data_dir,
            image_size=image_size,
            steps_per_epoch=steps_per_epoch,
            resize_mode=resize_mode,
            view="bscan",
            **kwargs,
        )

    def _process_index_file(self, df_index: pd.DataFrame) -> pd.DataFrame:
        data_dir = Path(self.hparams.data_dir)

        labels = sorted(df_index["diagnosis"].unique())
        self._label_to_idx = {label: idx for idx, label in enumerate(labels)}

        df_index["image"] = (data_dir / df_index["image_path"]).astype(str)
        df_index["label"] = df_index["diagnosis"].map(self._label_to_idx)
        df_index["group"] = df_index["subject_id"]
        return df_index[["image", "label", "group", "split"]]

    def _create_data_splits(self, df_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        columns = ["image", "label", "group"]
        df_train = df_index[df_index["split"] == "train"][columns].reset_index(drop=True)
        df_val = df_index[df_index["split"] == "test"][columns].reset_index(drop=True)
        df_test = df_val.iloc[:0]
        return df_train, df_val, df_test

    def _build_preprocess(self) -> list:
        return [
            mt.LoadImageD(keys=["image"], reader=ImageSeriesReader, dtype=None),
            *standardize(["image"], self.sample_depth, self.hparams.image_size, resize_mode=self.hparams.resize_mode),
        ]

    def train_dataloader(self):
        num_samples = self._steps_per_epoch() * self.hparams.batch_size
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            sampler=RandomSampler(self.train_dataset, replacement=True, num_samples=num_samples),
            collate_fn=self._collate,
            pin_memory=True,
            drop_last=True,
        )

    def _steps_per_epoch(self) -> int:
        if self.hparams.steps_per_epoch is not None:
            return self.hparams.steps_per_epoch

        world_size = self.trainer.world_size if self.trainer is not None else 1
        samples_per_rank = len(self.train_dataset) / world_size
        return max(1, math.ceil(samples_per_rank / self.hparams.batch_size))
