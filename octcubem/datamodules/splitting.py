from collections.abc import Sequence
from functools import reduce
from math import gcd

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedGroupKFold


def stratified_split(
    labels: npt.ArrayLike,
    fractions: Sequence[float],
    random_state: int | None = None,
) -> list[np.ndarray]:
    """Split sample indices into groups sized by `fractions`, preserving label proportions.

    Generalizes `sklearn.model_selection.train_test_split`'s `stratify` option to an
    arbitrary number of splits: each split gets roughly `fractions[i]` of every class's
    samples. `fractions` must sum to 1.

    Args:
        labels: array-like of per-sample class labels, length n_samples.
        fractions: sequence of fractions (summing to 1), one per split.
        random_state: seed for reproducible shuffling.

    Returns:
        A list of np.ndarray of indices into `labels`, one per entry in `fractions`.
    """
    fractions = list(fractions)
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)

    split_indices = [[] for _ in fractions]
    for label in np.unique(labels):
        group_idx = np.flatnonzero(labels == label)
        n = len(group_idx)
        shuffled = group_idx[rng.permutation(n)]

        counts = [round(n * fraction) for fraction in fractions]
        counts[-1] = n - sum(counts[:-1])

        start = 0
        for i, count in enumerate(counts):
            split_indices[i].extend(shuffled[start : start + count].tolist())
            start += count

    return [np.array(idx, dtype=int) for idx in split_indices]


def grouped_stratified_split(
    labels: npt.ArrayLike,
    groups: npt.ArrayLike,
    fractions: Sequence[float],
    random_state: int | None = None,
) -> list[np.ndarray]:
    """Split sample indices into groups sized by `fractions`, preserving label
    proportions while keeping every group (e.g. subject) entirely on one side
    of the split — so no group leaks across splits.

    Same indices-in/indices-out contract as `stratified_split`, but group-aware
    like `grouped_stratified_kfold`. Built on the same `StratifiedGroupKFold`:
    `fractions` are converted to integer shares (e.g. `[0.6, 0.2, 0.2]` ->
    `[3, 1, 1]`), that many folds are cut, and each split is handed its share
    of folds — reusing `StratifiedGroupKFold`'s group- and label-aware fold
    assignment rather than re-deriving it.

    Args:
        labels: array-like of per-sample class labels, length n_samples, or
            `(n_samples, n_labels)` multi-hot rows (see `stratification_key`).
        groups: array-like of per-sample group ids, length n_samples.
        fractions: sequence of fractions (summing to 1), one per split.
        random_state: seed for reproducible shuffling.

    Returns:
        A list of np.ndarray of indices into `labels`, one per entry in `fractions`.
    """
    fractions = list(fractions)
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    labels = stratification_key(labels)
    groups = np.asarray(groups)

    millishares = [round(fraction * 1000) for fraction in fractions]
    denom = reduce(gcd, millishares)
    shares = [milli // denom for milli in millishares]
    n_splits = sum(shares)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_val_indices = [val_idx for _, val_idx in splitter.split(np.zeros(len(labels)), labels, groups)]

    split_indices = []
    start = 0
    for share in shares:
        split_indices.append(np.concatenate(fold_val_indices[start : start + share]))
        start += share
    return split_indices


def grouped_stratified_kfold(
    labels: npt.ArrayLike,
    groups: npt.ArrayLike,
    n_splits: int,
    random_state: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Splits sample indices into `n_splits` (train_idx, val_idx) folds, keeping
    every group (e.g. subject) entirely on one side of each fold — so no
    subject leaks between train and val — while approximately preserving each
    fold's class balance.

    Thin wrapper around `sklearn.model_selection.StratifiedGroupKFold`, kept
    here for the same indices-in/indices-out contract as `stratified_split`.

    Args:
        labels: array-like of per-sample class labels, length n_samples, or
            `(n_samples, n_labels)` multi-hot rows (see `stratification_key`).
        groups: array-like of per-sample group ids, length n_samples.
        n_splits: number of folds.
        random_state: seed for reproducible fold assignment.

    Returns:
        A list of `(train_idx, val_idx)` index-array pairs, one per fold.
    """
    labels = stratification_key(labels)
    groups = np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


def stratification_key(labels: npt.ArrayLike) -> np.ndarray:
    """One stratification class per sample.

    Class labels pass through. Multi-hot rows become label-powerset ids -- each
    distinct combination of labels is one class -- since `StratifiedGroupKFold`
    stratifies on a single label. Its greedy assignment balances the frequent
    combinations; rare ones land wherever their group fits.
    """
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels
    _, combination_ids = np.unique(labels, axis=0, return_inverse=True)
    return combination_ids.reshape(-1)
