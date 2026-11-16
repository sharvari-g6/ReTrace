"""Attach the problem-grouped split (from data/make_splits.py) to a step-level matrix.

Run from repo root, after `python -m data.make_splits`:
    python -m training.attach_splits --matrix training/example_matrix.csv \
        --output training/split_matrix.csv
"""
import argparse
import pandas as pd
from features.schema import GROUP_COLUMN, SPLIT_COLUMN, TARGET_COLUMN
from features.validate_schema import validate_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", default="training/example_matrix.csv")
    p.add_argument("--splits", default="data/processed/sample_splits.csv",
                   help="sample_id, source, problem_id, split (from data.make_splits)")
    p.add_argument("--output", default="training/split_matrix.csv")
    p.add_argument("--no-signals", action="store_true",
                   help="matrix has no signal columns yet (e.g. labels only, for the step probe)")
    p.add_argument("--allow-partial-splits", action="store_true",
                   help="do not require train/val/test all present (tiny smoke tests only)")
    a = p.parse_args()

    df = pd.read_csv(a.matrix).drop(columns=[GROUP_COLUMN, SPLIT_COLUMN, "source"], errors="ignore")
    splits = pd.read_csv(a.splits)[["sample_id", "source", GROUP_COLUMN, SPLIT_COLUMN]]

    df = df.merge(splits, on="sample_id", how="left", validate="many_to_one")
    missing = df.loc[df[SPLIT_COLUMN].isna(), "sample_id"].unique()
    if len(missing):
        raise ValueError(f"{len(missing)} sample_ids missing from split table, e.g. {list(missing[:5])}")

    df = validate_matrix(df, require_signals=not a.no_signals,
                         require_all_splits=not a.allow_partial_splits)
    df.to_csv(a.output, index=False)

    for name in ("train", "val", "test"):
        part = df[df[SPLIT_COLUMN] == name]
        if part.empty:
            continue
        rate = part[TARGET_COLUMN].mean()
        print(f"{name:5s}: {part['sample_id'].nunique():6d} trajectories, "
              f"{part[GROUP_COLUMN].nunique():6d} problems, {len(part):7d} step rows, "
              f"error rate {rate:.3f}")
        if part[TARGET_COLUMN].nunique() < 2:
            print(f"  WARNING: split '{name}' contains only one class.")
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()
