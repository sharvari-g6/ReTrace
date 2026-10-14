"""Train the logistic fusion model over the four ReTrace signals."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from calibration.normalize import TrainOnlyNormalizer
from training.generate_features import SIGNAL_COLUMNS, validate_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CSV with true precomputed signals and fixed splits")
    p.add_argument("--model", default="artifacts/rfs_fusion.joblib")
    p.add_argument("--normalizer", default="artifacts/rfs_normalizer.json")
    p.add_argument("--metrics", default="artifacts/rfs_validation_metrics.json")
    args = p.parse_args()
    df = pd.read_csv(args.input)
    validate_features(df)
    normalizer = TrainOnlyNormalizer(SIGNAL_COLUMNS).fit(df)
    scaled = normalizer.transform(df)
    train = scaled[scaled.split == "train"]
    val = scaled[scaled.split == "validation"]
    if train.label.nunique() < 2:
        raise ValueError("Training labels must include both classes (0 and 1).")
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(train[SIGNAL_COLUMNS].to_numpy(), train.label.astype(int).to_numpy())
    probs = model.predict_proba(val[SIGNAL_COLUMNS].to_numpy())[:, 1]
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "validation_auroc": float(roc_auc_score(val.label, probs)) if val.label.nunique() == 2 else None,
        "validation_f1_at_0.5": float(f1_score(val.label, pred, zero_division=0)),
        "validation_precision_at_0.5": float(precision_score(val.label, pred, zero_division=0)),
        "validation_recall_at_0.5": float(recall_score(val.label, pred, zero_division=0)),
        "signal_columns": SIGNAL_COLUMNS,
        "coefficients": {c: float(v) for c, v in zip(SIGNAL_COLUMNS, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
    }
    bundle = {"model": model, "feature_columns": SIGNAL_COLUMNS,
              "label_convention": {"0": "correct", "1": "hallucinated/error"},
              "score_semantics": "predict_proba class 1 = estimated probability of hallucinated/error"}
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    normalizer.save(args.normalizer)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"RFS fusion model saved: {args.model}")
    print(f"Train-only normalizer saved: {args.normalizer}")
    print(json.dumps(metrics, indent=2))
    print("This is a learned fusion probability from the four signals; it is not yet a calibrated probability unless calibration is evaluated.")

if __name__ == "__main__":
    main()
