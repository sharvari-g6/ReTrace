"""Train a step-error probe using frozen hidden-state features (h_0000, ...)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from calibration.normalize import TrainOnlyNormalizer
from training.generate_features import validate_features


def safe_metrics(y, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    out = {"accuracy": float(accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0))}
    out["auroc"] = float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Validated CSV; includes h_0000, h_0001, ...")
    p.add_argument("--model", default="artifacts/step_probe.joblib")
    p.add_argument("--normalizer", default="artifacts/step_probe_normalizer.json")
    p.add_argument("--metrics", default="artifacts/step_probe_validation_metrics.json")
    args = p.parse_args()
    df = pd.read_csv(args.input)
    hidden = validate_features(df, require_hidden=True)
    normalizer = TrainOnlyNormalizer(hidden).fit(df)
    scaled = normalizer.transform(df)
    train = scaled[scaled.split == "train"]
    val = scaled[scaled.split == "validation"]
    if train.label.nunique() < 2:
        raise ValueError("Training labels must include both classes (0 and 1) for Logistic Regression.")
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(train[hidden].to_numpy(), train.label.astype(int).to_numpy())
    probs = model.predict_proba(val[hidden].to_numpy())[:, 1]
    metrics = safe_metrics(val.label.astype(int).to_numpy(), probs)
    bundle = {"model": model, "feature_columns": hidden,
              "label_convention": {"0": "correct", "1": "hallucinated/error"},
              "training_split_only": True}
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model)
    normalizer.save(args.normalizer)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Step-error probe trained using frozen hidden states.")
    print(f"Model: {args.model}\nNormalizer: {args.normalizer}\nValidation metrics: {json.dumps(metrics, indent=2)}")
    print("Test split was not used for fitting or model selection.")

if __name__ == "__main__":
    main()
