"""Validate and standardize the precomputed ReTrace feature table.

This module does NOT fabricate signals or extract Qwen hidden states. Supply actual
computed signals and, for the step probe, frozen hidden-state columns (h_0000, ...).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from calibration.normalize import normalize_splits

SIGNAL_COLUMNS = ["ars", "georeason", "prefix_hallucination", "causal_consistency"]
REQUIRED_COLUMNS = ["sample_id", "group_id", "split", "label", *SIGNAL_COLUMNS]
VALID_SPLITS = {"train", "validation", "test"}


def validate_features(frame: pd.DataFrame, require_hidden: bool = False) -> list[str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be present and unique for every row.")
    if frame[["group_id", "split", "label"]].isna().any().any():
        raise ValueError("group_id, split, and label cannot be missing.")
    unknown = set(frame["split"].astype(str).unique()) - VALID_SPLITS
    if unknown:
        raise ValueError(f"Unknown split value(s): {sorted(unknown)}. Use train/validation/test.")
    labels = set(pd.to_numeric(frame["label"], errors="raise").astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError("Label convention is 0=correct, 1=hallucinated/error.")
    if not (frame["label"].astype(float) == frame["label"].astype(int)).all():
        raise ValueError("Labels must be integer 0 or 1.")
    # The same parent question/trace must never cross splits.
    split_counts = frame.groupby("group_id")["split"].nunique()
    leaked_groups = split_counts[split_counts > 1].index.tolist()
    if leaked_groups:
        raise ValueError(f"group_id leakage across splits; examples: {leaked_groups[:5]}")
    numeric = list(SIGNAL_COLUMNS)
    hidden = sorted([c for c in frame.columns if c.startswith("h_")])
    if require_hidden and not hidden:
        raise ValueError("Step probe requires frozen hidden-state columns named h_0000, h_0001, ...")
    for col in numeric + hidden:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not frame[col].map(lambda x: pd.notna(x) and abs(float(x)) != float("inf")).all():
            raise ValueError(f"Column {col} contains NaN or infinite values.")
    if set(frame["split"].unique()) != VALID_SPLITS:
        raise ValueError("All three splits must be present: train, validation, and test.")
    return hidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing real/precomputed signals and fixed splits")
    parser.add_argument("--output", required=True, help="Validated feature CSV output path")
    parser.add_argument("--normalize", action="store_true", help="Also save train-only normalized signal columns")
    parser.add_argument("--normalizer", default="artifacts/rfs_normalizer.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    validate_features(frame)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.normalize:
        normalized, _ = normalize_splits(frame, SIGNAL_COLUMNS, args.normalizer)
        normalized.to_csv(args.output, index=False)
        print(f"Validated and normalized features saved to: {args.output}")
        print(f"Training-only normalizer saved to: {args.normalizer}")
    else:
        frame.to_csv(args.output, index=False)
        print(f"Validated features saved to: {args.output}")
    print("Label convention: 0=correct, 1=hallucinated/error")
    print(frame["split"].value_counts().reindex(["train", "validation", "test"]).to_string())

if __name__ == "__main__":
    main()
