"""Shared, deterministic cross-fitting folds (agree this with Nitish; both sides import it).

fold = int(problem_id, 16) % K, so every trajectory of a problem is in the same fold, the
assignment needs no extra file, and it does not change when data is added or removed.
Used for out-of-fold step-probe predictions (Consistency) and out-of-fold R0 (Deviation).
Only TRAIN rows are cross-fitted; val/test rows are scored by models fit on all of train.
"""
import numpy as np

N_FOLDS = 5


def fold_of(problem_id: str, k: int = N_FOLDS) -> int:
    return int(str(problem_id), 16) % k


def assign_folds(problem_ids, k: int = N_FOLDS) -> np.ndarray:
    return np.array([fold_of(p, k) for p in problem_ids])
