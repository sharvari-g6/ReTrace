"""Merge per-signal CSVs into the four-signal feature matrix and validate it.

Every signal file uses the same key (sample_id, step_id; step_id zero-based) and has
exactly one signal column named as in features/schema.py, e.g.
    features/ars_scores.csv          sample_id, step_id, ARS
    features/deviation_scores.csv    sample_id, step_id, Deviation
    features/consistency_scores.csv  sample_id, step_id, Consistency
    features/causal_scores.csv       sample_id, step_id, Causal

Run from repo root:
    python -m training.generate_features --labels training/labels.csv \
        --signal features/ars_scores.csv --signal features/deviation_scores.csv \
        --signal features/consistency_scores.csv --signal features/causal_scores.csv \
        --output training/split_matrix.csv

--labels is a CSV with sample_id, step_id, label and (problem_id, split) columns.
This script does NOT invent signal values; missing rows are an error.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from features.schema import KEY_COLUMNS, SIGNAL_COLUMNS, TARGET_COLUMN, MATRIX_COLUMNS
from features.validate_schema import validate_matrix


def merge_signals(base: pd.DataFrame, signal_frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    n = len(out)
    for f in signal_frames:
        cols = [c for c in f.columns if c not in KEY_COLUMNS]
        if len(cols) != 1 or cols[0] not in SIGNAL_COLUMNS:
            raise ValueError(f"Signal file must have exactly one of {SIGNAL_COLUMNS}, got {cols}")
        out = out.merge(f, on=KEY_COLUMNS, how="left", validate="one_to_one")
        if len(out) != n:
            raise ValueError("Merge changed the row count.")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--signal", action="append", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    base = pd.read_csv(a.labels)
    frames = [pd.read_csv(s) for s in a.signal]
    df = merge_signals(base, frames)
    df = validate_matrix(df)          # fails on NaN (missing signal rows), leakage, bad steps
    df = df[[c for c in MATRIX_COLUMNS if c in df.columns]]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(f"Saved {a.output}: {len(df)} rows")
    print("Label balance:", df[TARGET_COLUMN].value_counts().to_dict())
    print(df.groupby("split").size().to_string())


if __name__ == "__main__":
    main()
