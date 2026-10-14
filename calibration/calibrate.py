"""Choose threshold on validation, freeze it, then evaluate test once."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from calibration.normalize import TrainOnlyNormalizer
from calibration.threshold import choose_threshold
from training.generate_features import SIGNAL_COLUMNS, validate_features


def evaluate(y, probs, threshold):
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(probs) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix_labels_0_1": cm,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Same fixed-split CSV used to train RFS model")
    p.add_argument("--model", default="artifacts/rfs_fusion.joblib")
    p.add_argument("--normalizer", default="artifacts/rfs_normalizer.json")
    p.add_argument("--output", default="artifacts/rfs_calibration_report.json")
    args = p.parse_args()
    df = pd.read_csv(args.input)
    validate_features(df)
    bundle = joblib.load(args.model)
    cols = bundle["feature_columns"]
    if cols != SIGNAL_COLUMNS:
        raise ValueError(f"Expected RFS signal columns {SIGNAL_COLUMNS}, found {cols}")
    normalizer = TrainOnlyNormalizer.load(args.normalizer)
    if normalizer.columns != cols:
        raise ValueError("Normalizer columns do not match model feature columns.")
    scaled = normalizer.transform(df)
    model = bundle["model"]
    val = scaled[scaled.split == "validation"]
    test = scaled[scaled.split == "test"]
    val_probs = model.predict_proba(val[cols].to_numpy())[:, 1]
    chosen = choose_threshold(val.label.astype(int).to_numpy(), val_probs)
    # The threshold is now frozen. Only now are test predictions and metrics produced.
    test_probs = model.predict_proba(test[cols].to_numpy())[:, 1]
    report = {
        "label_convention": {"0": "correct", "1": "hallucinated/error"},
        "threshold_selection_split": "validation",
        "test_used_for_threshold_selection": False,
        "validation_threshold_metrics": chosen,
        "test_metrics_at_frozen_threshold": evaluate(test.label, test_probs, chosen["threshold"]),
        "warning": "Use the test result as a final holdout estimate; do not tune based on it.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved: {args.output}")

if __name__ == "__main__":
    main()
