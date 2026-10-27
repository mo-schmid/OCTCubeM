import lightning.pytorch as pl
from lightning.pytorch.utilities import CombinedLoader

from octcubem.datamodules.alleye_macula import AlleyeMaculaDataModule
from octcubem.datamodules.alleye_macula_bscans import AlleyeMaculaBscanDataModule
from octcubem.datamodules.kermany import Kermany2018DataModule


class JointPretrainDataModule(pl.LightningDataModule):
    """OCTCube's joint 3D + 2D pretraining data: every training step pairs a batch of
    volumes with a batch of `batch_size_2d` single B-scans.

    Wraps one datamodule per source and runs their train loaders in lockstep: all are
    cut to the volume stream's epoch length, and a training batch is the dict
    `{"3d": ..., "2d_alleye": ..., "2d_kermany": ...}` of whichever sources are set.
    The 2D batch is split between the two 2D sources at a fixed ratio, the Kermany
    share being `kermany_fraction` -- by default its share of the pooled B-scans, which
    is what the original's uniform sampling over the concatenated datasets amounts to.

    Validation runs one loader per source, in the order of `val_names`; the
    2D ones batch `batch_size_2d` B-scans.

    The batch sizes the wrapped datamodules were created with are overridden: the
    volume batch size is kept, the 2D ones are set from `batch_size_2d`. The split is
    recomputed whenever the train loader is built, as the alleye pool shrinks under
    self-paced learning; `hardness` is the alleye datamodule's `HardnessTable` then.
    """

    def __init__(
        self,
        volumes: AlleyeMaculaDataModule,
        bscans: AlleyeMaculaBscanDataModule | None = None,
        kermany: Kermany2018DataModule | None = None,
        batch_size_2d: int = 64,
        kermany_fraction: float | None = None,
    ):
        super().__init__()
        if bscans is None and kermany is None:
            raise ValueError("Joint pretraining needs at least one 2D source")
        self.save_hyperparameters(ignore=["volumes", "bscans", "kermany"])
        self.sources = {"3d": volumes, "2d_alleye": bscans, "2d_kermany": kermany}
        self.sources = {name: dm for name, dm in self.sources.items() if dm is not None}
        self.train_batch_sizes_2d = None

    @property
    def hardness(self):
        """The alleye B-scans' `HardnessTable` under self-paced learning, else `None`."""
        bscans = self.sources.get("2d_alleye")
        return None if bscans is None else bscans.hardness

    @property
    def val_names(self) -> list[str]:
        """The source each val loader reads, by `dataloader_idx`."""
        return list(self.sources)

    def setup(self, stage=None):
        for dm in self.sources.values():
            # the wrapped datamodules size their epochs from the trainer's world size
            dm.trainer = self.trainer
            dm.setup(stage)

    def _split_batch_size_2d(self) -> dict[str, int]:
        batch_size = self.hparams.batch_size_2d
        if "2d_kermany" not in self.sources:
            return {"2d_alleye": batch_size}
        if "2d_alleye" not in self.sources:
            return {"2d_kermany": batch_size}

        fraction = self.hparams.kermany_fraction
        if fraction is None:
            num_kermany = len(self.sources["2d_kermany"].train_dataset)
            num_bscans = self.sources["2d_alleye"].num_train_bscans
            fraction = num_kermany / (num_kermany + num_bscans)
        # both sources stay in every batch, however small the batch
        num_kermany = min(max(round(batch_size * fraction), 1), batch_size - 1)
        return {"2d_alleye": batch_size - num_kermany, "2d_kermany": num_kermany}

    def train_dataloader(self):
        steps = self.sources["3d"]._steps_per_epoch()
        self.train_batch_sizes_2d = self._split_batch_size_2d()
        loaders = {"3d": self.sources["3d"].train_dataloader()}
        for name, batch_size in self.train_batch_sizes_2d.items():
            dm = self.sources[name]
            dm.hparams.steps_per_epoch = steps
            loaders[name] = _with_batch_size(dm, batch_size, dm.train_dataloader)
        return CombinedLoader(loaders, mode="min_size")

    def val_dataloader(self):
        loaders = [self.sources["3d"].val_dataloader()]
        for name in self.val_names[1:]:
            dm = self.sources[name]
            loaders.append(_with_batch_size(dm, self.hparams.batch_size_2d, dm.val_dataloader))
        return loaders

    def state_dict(self) -> dict:
        return {name: dm.state_dict() for name, dm in self.sources.items()}

    def load_state_dict(self, state_dict: dict):
        for name, state in state_dict.items():
            self.sources[name].load_state_dict(state)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        # a training batch holds one batch per source, a val batch is one source's
        if isinstance(batch, dict) and set(batch) <= set(self.sources):
            return {name: self.sources[name].on_after_batch_transfer(part, 0) for name, part in batch.items()}
        dm = self.sources[self.val_names[dataloader_idx]]
        return dm.on_after_batch_transfer(batch, 0)


def _with_batch_size(dm: pl.LightningDataModule, batch_size: int, build_loader):
    """Builds a loader of `dm` with `batch_size`; a loader keeps the batch size it was built with."""
    dm.hparams.batch_size = batch_size
    return build_loader()
