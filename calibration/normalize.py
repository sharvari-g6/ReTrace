"""Leakage-safe standardization: fit statistics on TRAIN rows only."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


class TrainOnlyNormalizer:
    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "TrainOnlyNormalizer":
        if "split" not in frame:
            raise ValueError("Input must contain a 'split' column.")
        train = frame.loc[frame["split"].eq("train"), self.columns]
        if train.empty:
            raise ValueError("Cannot fit normalizer: no rows have split='train'.")
        values = train.astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Training features contain NaN or infinite values.")
        self.mean_ = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        # Constant columns should remain finite after scaling.
        self.scale_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() on training rows before transform().")
        result = frame.copy()
        values = result[self.columns].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Features contain NaN or infinite values.")
        result.loc[:, self.columns] = (values - self.mean_) / self.scale_
        return result

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Normalizer has not been fitted.")
        payload = {
            "fit_split": "train",
            "columns": self.columns,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainOnlyNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("fit_split") != "train":
            raise ValueError("Normalizer metadata does not confirm training-only fit.")
        obj = cls(payload["columns"])
        obj.mean_ = np.asarray(payload["mean"], dtype=float)
        obj.scale_ = np.asarray(payload["scale"], dtype=float)
        return obj


def normalize_splits(frame: pd.DataFrame, columns: Iterable[str], output_json: str | Path | None = None):
    """Fit on train only, transform train/validation/test, return (frame, normalizer)."""
    normalizer = TrainOnlyNormalizer(columns).fit(frame)
    normalized = normalizer.transform(frame)
    if output_json is not None:
        normalizer.save(output_json)
    return normalized, normalizer
