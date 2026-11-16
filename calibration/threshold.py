"""Select a binary decision threshold using VALIDATION rows only."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def choose_threshold(y_true, probabilities, thresholds=None) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("y_true and probabilities must have equal non-zero lengths.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels must be 0=correct, 1=hallucinated/error.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        rows.append({"threshold": float(t),
                     "f1": float(f1_score(y, pred, zero_division=0)),
                     "precision": float(precision_score(y, pred, zero_division=0)),
                     "recall": float(recall_score(y, pred, zero_division=0))})
    # Max F1, then precision, then closest to 0.5 for deterministic tie-breaking.
    best = sorted(rows, key=lambda r: (-r["f1"], -r["precision"], abs(r["threshold"] - 0.5)))[0]
    return best
