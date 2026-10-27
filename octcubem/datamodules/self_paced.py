"""Self-paced learning (SPL) for OCTCube's 2D branch.

OCTCube scores every B-scan of the 3D train volumes by the 3D model's reconstruction
loss of the temporal patch holding it (its "hardness"), and after every epoch narrows
the 2D B-scan pool to the hardest `K` fraction -- a random `K` fraction in the first
epoch, before anything is scored. `HardnessTable` keeps those scores and the current
selection; `slice_losses` maps the model's per-patch losses back to stored B-scans.
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch


class HardnessTable:
    """Hardness of every B-scan of a fixed set of volumes, and the B-scans selected for training.

    Rows are volumes (by `sample_id`), columns their stored B-scans; a volume with
    fewer than the most B-scans leaves the rest of its row unused. Hardness starts at 0,
    as in the original. Updates of one epoch are collected per process and only
    applied by `merge`, which averages them across processes.

    Args:
        sample_ids: The volumes, in row order.
        num_slices: How many B-scans each volume stores.
    """

    def __init__(self, sample_ids: Sequence[str], num_slices: Sequence[int]):
        self.sample_ids = list(sample_ids)
        self.num_slices = np.asarray(num_slices, dtype=np.int64)
        self.rows = {sample_id: row for row, sample_id in enumerate(self.sample_ids)}
        shape = (len(self.sample_ids), int(self.num_slices.max()))
        self.valid = np.arange(shape[1])[None, :] < self.num_slices[:, None]
        self.hardness = np.zeros(shape, dtype=np.float32)
        self.selection = self.valid.copy()
        self._pending = np.zeros(shape, dtype=np.float32)
        self._pending_mask = np.zeros(shape, dtype=bool)

    @property
    def num_bscans(self) -> int:
        return int(self.valid.sum())

    @property
    def num_selected(self) -> int:
        return int(self.selection.sum())

    def update(self, sample_ids: Sequence[str], losses: Sequence[np.ndarray]):
        """Records the hardness of every B-scan of `sample_ids`, one `num_slices`-long array each.

        Volumes outside the table are skipped. A volume seen twice before `merge` keeps
        the later losses, as the original's overwrite does.
        """
        for sample_id, loss in zip(sample_ids, losses):
            row = self.rows.get(sample_id)
            if row is None:
                continue
            self._pending[row, : len(loss)] = loss
            self._pending_mask[row, : len(loss)] = True

    def merge(self, reduce_sum: Callable[[torch.Tensor], torch.Tensor] | None = None):
        """Applies the pending updates, averaged over the processes that made one.

        `reduce_sum` sums a tensor across processes (e.g. `strategy.reduce(t, reduce_op="sum")`);
        `None` for a single process. B-scans no process scored keep their hardness.
        """
        pending = torch.from_numpy(self._pending * self._pending_mask)
        counts = torch.from_numpy(self._pending_mask.astype(np.float32))
        if reduce_sum is not None:
            pending, counts = reduce_sum(pending), reduce_sum(counts)
        pending, counts = pending.cpu().numpy(), counts.cpu().numpy()
        scored = counts > 0
        self.hardness[scored] = pending[scored] / counts[scored]
        self._pending[:] = 0
        self._pending_mask[:] = False

    def select_random(self, fraction: float, seed: int = 0):
        """Selects a random `fraction` of all B-scans, the same for every process given `seed`."""
        num = int(fraction * self.num_bscans)
        flat = np.flatnonzero(self.valid)
        chosen = np.random.default_rng(seed).choice(flat, num, replace=False)
        self._select_flat(chosen)

    def select_hardest(self, fraction: float):
        """Selects the hardest `fraction` of all B-scans."""
        num = int(fraction * self.num_bscans)
        flat = np.flatnonzero(self.valid)
        hardness = self.hardness.reshape(-1)[flat]
        # a stable sort, so equal hardness (e.g. the unscored 0s) breaks ties the same way everywhere
        order = np.argsort(-hardness, kind="stable")
        self._select_flat(flat[order[:num]])

    def _select_flat(self, flat_indices: np.ndarray):
        selection = np.zeros(self.valid.size, dtype=bool)
        selection[flat_indices] = True
        self.selection = selection.reshape(self.valid.shape)

    def state_dict(self) -> dict:
        # tensors rather than arrays: a Lightning checkpoint loads with `weights_only=True`
        return {
            "sample_ids": self.sample_ids,
            "hardness": torch.from_numpy(self.hardness.copy()),
            "selection": torch.from_numpy(self.selection.copy()),
        }

    def load_state_dict(self, state: dict):
        if list(state["sample_ids"]) != self.sample_ids:
            raise ValueError("The saved hardness table covers different volumes than this dataset")
        self.hardness = state["hardness"].numpy().astype(np.float32)
        self.selection = state["selection"].numpy().astype(bool)


def slice_losses(
    patch_losses: np.ndarray,
    depth_flipped: bool,
    num_slices: int,
    depth: int,
    t_patch_size: int,
) -> np.ndarray:
    """The loss of every stored B-scan of a volume, from the losses of its temporal patches.

    The model saw the volume centre-cropped or zero-padded to `depth` B-scans (as
    `standardize` does), possibly reversed in depth, in temporal patches of
    `t_patch_size`; `patch_losses` holds one loss per patch, in model order. A stored
    B-scan takes the loss of the patch it ended up in; one cropped away takes the loss
    of the nearest patch, as the original gives its leftover last B-scan the last loss.
    """
    # where `ResizeWithPadOrCrop` puts stored B-scan 0, as a cropped index
    offset = num_slices // 2 - depth // 2 if num_slices >= depth else -((depth - num_slices) // 2)
    cropped = np.clip(np.arange(num_slices) - offset, 0, depth - 1)
    model_index = depth - 1 - cropped if depth_flipped else cropped
    return np.asarray(patch_losses)[model_index // t_patch_size]
