"""Validate a ReTrace feature matrix. Raises ValueError (not assert) on any problem."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from features.schema import (KEY_COLUMNS, SIGNAL_COLUMNS, TARGET_COLUMN,
                             GROUP_COLUMN, SPLIT_COLUMN, VALID_SPLITS, VERIFIED_COLUMN)


def validate_matrix(df: pd.DataFrame, *, require_signals: bool = True,
                    require_split: bool = True, require_all_splits: bool = True,
                    require_verified: bool = False) -> pd.DataFrame:
    """Return a cleaned copy (numeric dtypes enforced). Does not mutate the input.

    require_signals=False is the mode for step-probe training, which runs BEFORE
    the Consistency/Deviation signals exist.
    """
    df = df.copy()
    need = KEY_COLUMNS + [TARGET_COLUMN]
    if require_split:
        need += [GROUP_COLUMN, SPLIT_COLUMN]
    if require_signals:
        need += SIGNAL_COLUMNS
    if require_verified:
        need += [VERIFIED_COLUMN]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[need].isna().any().any():
        bad = df[need].isna().sum()
        raise ValueError(f"NaN in required columns: {bad[bad > 0].to_dict()}")

    if df.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate (sample_id, step_id) rows found.")

    df["step_id"] = pd.to_numeric(df["step_id"], errors="raise").astype(int)
    g = df.groupby("sample_id")["step_id"]
    if (g.min() != 0).any():
        raise ValueError("step_id must start at 0 for every sample (zero-based).")
    if ((g.max() + 1) != g.count()).any():
        raise ValueError("step_id must be contiguous (0..n-1) within each sample.")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="raise")
    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("label must be 0 (correct) or 1 (error).")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    if VERIFIED_COLUMN in df.columns:
        df[VERIFIED_COLUMN] = pd.to_numeric(df[VERIFIED_COLUMN], errors="raise")
        if not df[VERIFIED_COLUMN].isin([0, 1]).all():
            raise ValueError("verified must be 0 or 1.")
        df[VERIFIED_COLUMN] = df[VERIFIED_COLUMN].astype(int)
        # verified steps must be a prefix of each trajectory; an error label there can only be the last one
        v = df[df[VERIFIED_COLUMN] == 1]
        if (v.groupby("sample_id")[TARGET_COLUMN].sum() > 1).any():
            raise ValueError("A trajectory has more than one verified error step.")
    if require_signals:
        for c in SIGNAL_COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
            if not np.isfinite(df[c].to_numpy()).all():
                raise ValueError(f"Column {c} contains NaN or infinite values.")

    if require_split:
        unknown = set(df[SPLIT_COLUMN].unique()) - set(VALID_SPLITS)
        if unknown:
            raise ValueError(f"Unknown split values {sorted(unknown)}; use {VALID_SPLITS}.")
        if (df.groupby("sample_id")[SPLIT_COLUMN].nunique() > 1).any():
            raise ValueError("A sample_id appears in more than one split.")
        leaked = df.groupby(GROUP_COLUMN)[SPLIT_COLUMN].nunique()
        leaked = leaked[leaked > 1].index.tolist()
        if leaked:
            raise ValueError(f"{GROUP_COLUMN} leakage across splits, e.g. {leaked[:5]}")
        if require_all_splits and set(df[SPLIT_COLUMN].unique()) != set(VALID_SPLITS):
            raise ValueError(f"All splits {VALID_SPLITS} must be present.")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="training/split_matrix.csv")
    p.add_argument("--no-signals", action="store_true", help="skip signal checks (probe stage)")
    p.add_argument("--no-split", action="store_true", help="skip split checks (raw matrix)")
    a = p.parse_args()
    df = validate_matrix(pd.read_csv(a.input), require_signals=not a.no_signals,
                         require_split=not a.no_split)
    print("Schema validation successful.")
    print("Rows:", len(df), "| samples:", df["sample_id"].nunique())
    print("Label balance:", df[TARGET_COLUMN].value_counts().to_dict())
    if not a.no_split:
        print(df.groupby(SPLIT_COLUMN)[TARGET_COLUMN].agg(["count", "mean"]).rename(
            columns={"mean": "error_rate"}).to_string())


if __name__ == "__main__":
    main()
