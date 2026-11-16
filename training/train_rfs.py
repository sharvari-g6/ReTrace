"""Train the logistic fusion model (RFS) over the four ReTrace signals.

Run from repo root: python -m training.train_rfs --input training/split_matrix.csv
Ablations: --signals ARS Deviation   (any subset of the four signal columns)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from calibration.normalize import TrainOnlyNormalizer
from features.schema import SIGNAL_COLUMNS, TARGET_COLUMN, SPLIT_COLUMN, VERIFIED_COLUMN
from features.validate_schema import validate_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="split matrix with all four signals")
    p.add_argument("--signals", nargs="+", default=SIGNAL_COLUMNS, choices=SIGNAL_COLUMNS)
    p.add_argument("--class-weight", default="balanced", choices=["balanced", "none"],
                   help="team decision: balanced. Probabilities are then NOT calibrated: use Day 6 calibration")
    p.add_argument("--model", default="artifacts/rfs_fusion.joblib")
    p.add_argument("--normalizer", default="artifacts/rfs_normalizer.json")
    p.add_argument("--metrics", default="artifacts/rfs_validation_metrics.json")
    a = p.parse_args()
    cols = list(a.signals)

    df = validate_matrix(pd.read_csv(a.input), require_verified=True)
    normalizer = TrainOnlyNormalizer(cols).fit(df)          # train rows only
    scaled = normalizer.transform(df)
    train = scaled[scaled[SPLIT_COLUMN] == "train"]
    val_all = scaled[scaled[SPLIT_COLUMN] == "val"]
    val = val_all[val_all[VERIFIED_COLUMN] == 1]          # headline evaluation: verified steps only
    if train[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training labels must include both classes.")

    model = LogisticRegression(max_iter=2000, class_weight=(None if a.class_weight == 'none' else 'balanced'), random_state=42)
    model.fit(train[cols].to_numpy(), train[TARGET_COLUMN].to_numpy())
    yv = val[TARGET_COLUMN].to_numpy()
    probs = model.predict_proba(val[cols].to_numpy())[:, 1]
    pred = (probs >= 0.5).astype(int)
    two = len(set(yv)) == 2
    metrics = {
        "signals": cols,
        "evaluation_rows": "validation, verified steps only",
        "n_validation_rows": int(len(yv)), "n_validation_positives": int(yv.sum()),
        "class_weight": a.class_weight,
        "validation_auroc": float(roc_auc_score(yv, probs)) if two else None,
        "validation_pr_auc": float(average_precision_score(yv, probs)) if two else None,
        "validation_f1_at_0.5": float(f1_score(yv, pred, zero_division=0)),
        "validation_precision_at_0.5": float(precision_score(yv, pred, zero_division=0)),
        "validation_recall_at_0.5": float(recall_score(yv, pred, zero_division=0)),
        # baselines for RQ2 (all signals oriented higher = more error-like)
        "single_signal_validation_auroc": {c: float(roc_auc_score(yv, val[c])) for c in cols} if two else None,
        "equal_weight_validation_auroc": float(roc_auc_score(yv, val[cols].mean(axis=1))) if two else None,
        "coefficients": {c: float(v) for c, v in zip(cols, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
    }
    bundle = {"model": model, "feature_columns": cols,
              "label_convention": {"0": "correct", "1": "hallucinated/error"},
              "score_semantics": "predict_proba class 1 = estimated probability of error"}
    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, a.model)
    normalizer.save(a.normalizer)
    Path(a.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Not a calibrated probability until calibration is evaluated (Day 6).")


if __name__ == "__main__":
    main()
