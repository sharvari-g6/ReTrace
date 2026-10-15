"""Train the step-error probe on frozen hidden states (Member A).

Inputs
  --matrix     CSV with sample_id, step_id, label, problem_id, split  (signals NOT required;
               produce it with `python -m training.split_data --no-signals ...`)
  --hidden-dir directory of .pt shards. Each shard is one dict or a list of dicts with
               "sample_id" and "step_hidden_states" (tensor [n_steps, d]); row i = step_id i.
               (Format of Shaishavi's smoke_test.pt. Verify it against the real zip.)

Outputs
  artifacts/step_probe.joblib             final probe + scaler (fit on TRAIN only)
  artifacts/step_probe_validation_metrics.json
  features/step_probe_probs.csv           sample_id, step_id, probe_prob, origin
        origin = "oof"   : train rows, predicted by grouped K-fold (probe never saw that problem)
        origin = "final" : val/test rows, predicted by the final probe
  Use this file to build Consistency, so train-row Consistency is not overfit.
Test labels are never used here (test rows are only scored, never evaluated).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from features.schema import KEY_COLUMNS, TARGET_COLUMN, GROUP_COLUMN, SPLIT_COLUMN
from features.validate_schema import validate_matrix


def load_hidden(hidden_dir):
    import torch
    keys, chunks = [], []
    for path in sorted(Path(hidden_dir).glob("*.pt")):
        obj = torch.load(path, map_location="cpu")
        for item in (obj if isinstance(obj, list) else [obj]):
            H = item["step_hidden_states"].float().numpy()
            keys.append(pd.DataFrame({"sample_id": item["sample_id"],
                                      "step_id": np.arange(len(H))}))
            chunks.append(H)
    if not chunks:
        raise ValueError(f"No .pt shards found in {hidden_dir}")
    return pd.concat(keys, ignore_index=True), np.vstack(chunks)


def fit_probe(X, y, C):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def predict(scaler, clf, X):
    return clf.predict_proba(scaler.transform(X))[:, 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", required=True)
    p.add_argument("--hidden-dir", required=True)
    p.add_argument("--model", default="artifacts/step_probe.joblib")
    p.add_argument("--metrics", default="artifacts/step_probe_validation_metrics.json")
    p.add_argument("--probs", default="features/step_probe_probs.csv")
    p.add_argument("--C-grid", type=float, nargs="+", default=[0.0005, 0.005, 0.05, 0.5])
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()

    df = validate_matrix(pd.read_csv(a.matrix), require_signals=False)
    hkeys, H = load_hidden(a.hidden_dir)
    hkeys["row"] = np.arange(len(hkeys))
    df = df.merge(hkeys, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if df["row"].isna().any():
        raise ValueError(f"{int(df['row'].isna().sum())} matrix rows have no hidden state. "
                         "Check sample_id/step_id alignment (step_id zero-based).")
    X = H[df["row"].astype(int).to_numpy()]
    y = df[TARGET_COLUMN].to_numpy()
    groups = df[GROUP_COLUMN].to_numpy()
    tr = (df[SPLIT_COLUMN] == "train").to_numpy()
    va = (df[SPLIT_COLUMN] == "val").to_numpy()
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
        raise ValueError("Train and val must each contain both classes.")

    # choose C on VALIDATION only
    best = None
    for C in a.C_grid:
        sc, clf = fit_probe(X[tr], y[tr], C)
        auc = roc_auc_score(y[va], predict(sc, clf, X[va]))
        print(f"C={C:<8g} val AUROC={auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, C)
    _, C = best

    scaler, clf = fit_probe(X[tr], y[tr], C)
    pv = predict(scaler, clf, X[va])
    pred = (pv >= 0.5).astype(int)
    metrics = {"C": C, "validation_auroc": float(roc_auc_score(y[va], pv)),
               "validation_f1_at_0.5": float(f1_score(y[va], pred, zero_division=0)),
               "validation_precision_at_0.5": float(precision_score(y[va], pred, zero_division=0)),
               "validation_recall_at_0.5": float(recall_score(y[va], pred, zero_division=0)),
               "n_train_steps": int(tr.sum()), "n_val_steps": int(va.sum())}

    # probabilities for every row: OOF for train, final probe for val/test
    probs = np.zeros(len(df))
    origin = np.where(tr, "oof", "final").astype(object)
    idx_tr = np.where(tr)[0]
    n_splits = min(a.folds, len(np.unique(groups[idx_tr])))
    for f_tr, f_te in GroupKFold(n_splits=n_splits).split(idx_tr, groups=groups[idx_tr]):
        a_idx, b_idx = idx_tr[f_tr], idx_tr[f_te]
        if len(np.unique(y[a_idx])) < 2:
            raise ValueError("A CV fold has a single class; use fewer folds or more data.")
        sc, c = fit_probe(X[a_idx], y[a_idx], C)
        probs[b_idx] = predict(sc, c, X[b_idx])
    rest = np.where(~tr)[0]
    probs[rest] = predict(scaler, clf, X[rest])

    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler, "C": C,
                 "label_convention": {"0": "correct", "1": "error"}}, a.model)
    Path(a.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Path(a.probs).parent.mkdir(parents=True, exist_ok=True)
    out = df[KEY_COLUMNS].copy()
    out["probe_prob"] = probs
    out["origin"] = origin
    out.to_csv(a.probs, index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Saved {a.model}, {a.metrics}, {a.probs}")


if __name__ == "__main__":
    main()
