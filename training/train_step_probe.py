"""Train the step-error probe on frozen hidden states (Member A).

Inputs
  --matrix     CSV with sample_id, step_id, label, problem_id, split  (signals NOT required;
               produce it with `python -m training.split_data --no-signals ...`)
  --hidden-dir folder searched RECURSIVELY for *.pt shards, e.g. train_features_raw/{processbench,prm800k,halueval}/
               (equal file names in different subfolders are fine). Shaishavi's format:
               {"source": ..., "records": [ {"sample_id", ...,
                 "step_hidden_states": {layer: {"mean"|"last_token": tensor[n_steps, 4096]}}} ]}
               Row i of the tensor = step_id i. Choose the representation with --layer/--pooling.

Outputs
  artifacts/step_probe.joblib             final probe + scaler (fit on TRAIN only)
  artifacts/step_probe_validation_metrics.json
  features/step_probe_probs.csv           sample_id, step_id, probe_prob, origin
        origin = "oof"   : train rows, predicted out-of-fold with the shared problem folds
                         (features/folds.py), so the probe never saw that problem
        origin = "final" : val/test rows, predicted by the final probe
  Use this file to build Consistency, so train-row Consistency is not overfit.
Training uses ALL rows (post-error steps keep label 1). Headline metrics and the choice of C use
verified rows only (verified == 1) when that column exists.
Test labels are never used here (test rows are only scored, never evaluated).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from features.folds import assign_folds, N_FOLDS
from features.schema import KEY_COLUMNS, TARGET_COLUMN, GROUP_COLUMN, SPLIT_COLUMN, VERIFIED_COLUMN
from features.validate_schema import validate_matrix


def load_hidden(hidden_dir, layer=-1, pooling="mean", wanted=None):
    import torch
    keys, chunks = [], []
    for path in sorted(Path(hidden_dir).rglob("*.pt")):   # searches subfolders (same file names allowed per dataset)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        for item in obj["records"]:
            if wanted is not None and item["sample_id"] not in wanted:
                continue
            H = item["step_hidden_states"][layer][pooling].float().numpy()
            keys.append(pd.DataFrame({"sample_id": item["sample_id"],
                                      "step_id": np.arange(len(H))}))
            chunks.append(H)
    if not chunks:
        raise ValueError(f"No .pt shards found in {hidden_dir}")
    return pd.concat(keys, ignore_index=True), np.vstack(chunks)


def fit_probe(X, y, C, class_weight=None):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight=class_weight, random_state=42)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def predict(scaler, clf, X):
    return clf.predict_proba(scaler.transform(X))[:, 1]


def eval_block(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"n": int(len(y)), "positives": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            f"f1_at_{thr}": float(f1_score(y, pred, zero_division=0)),
            f"precision_at_{thr}": float(precision_score(y, pred, zero_division=0)),
            f"recall_at_{thr}": float(recall_score(y, pred, zero_division=0))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", required=True)
    p.add_argument("--hidden-dir", required=True)
    p.add_argument("--model", default="artifacts/step_probe.joblib")
    p.add_argument("--metrics", default="artifacts/step_probe_validation_metrics.json")
    p.add_argument("--probs", default="features/step_probe_probs.csv")
    p.add_argument("--layer", type=int, default=-1, choices=[-8, -4, -1])
    p.add_argument("--pooling", default="mean", choices=["mean", "last_token"])
    p.add_argument("--class-weight", default="balanced", choices=["balanced", "none"],
                   help="team decision: balanced. Probabilities are then NOT calibrated (Day 6).")
    p.add_argument("--C-grid", type=float, nargs="+", default=[0.00001, 0.0001, 0.0005, 0.005, 0.05])
    p.add_argument("--folds", type=int, default=N_FOLDS)
    a = p.parse_args()
    cw = None if a.class_weight == "none" else "balanced"

    df = validate_matrix(pd.read_csv(a.matrix), require_signals=False, require_all_splits=False)
    df_ids = set(df["sample_id"])
    hkeys, H = load_hidden(a.hidden_dir, a.layer, a.pooling, wanted=df_ids)
    hkeys["row"] = np.arange(len(hkeys))
    df = df.merge(hkeys, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if df["row"].isna().any():
        raise ValueError(f"{int(df['row'].isna().sum())} matrix rows have no hidden state. "
                         "Check sample_id/step_id alignment (step_id zero-based).")
    X = H[df["row"].astype(int).to_numpy()]
    y = df[TARGET_COLUMN].to_numpy()
    tr = (df[SPLIT_COLUMN] == "train").to_numpy()
    va = (df[SPLIT_COLUMN] == "val").to_numpy()
    if VERIFIED_COLUMN in df.columns:
        ver = df[VERIFIED_COLUMN].to_numpy() == 1
    else:
        print("WARNING: no 'verified' column; evaluating on all rows.")
        ver = np.ones(len(df), dtype=bool)
    va_ev = va & ver                                   # headline evaluation rows
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va_ev])) < 2:
        raise ValueError("Train and verified-val must each contain both classes.")

    # choose C on VERIFIED VALIDATION rows only
    best = None
    for C in a.C_grid:
        sc, clf = fit_probe(X[tr], y[tr], C, cw)
        auc = roc_auc_score(y[va_ev], predict(sc, clf, X[va_ev]))
        print(f"C={C:<8g} verified-val AUROC={auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, C)
    _, C = best

    scaler, clf = fit_probe(X[tr], y[tr], C, cw)
    pv = predict(scaler, clf, X[va])
    pv_ev = predict(scaler, clf, X[va_ev])
    # step-index-only baseline (absolute index; relative position would leak the label)
    pos = df[["step_id"]].to_numpy()
    pos_clf = LogisticRegression(max_iter=1000).fit(pos[tr], y[tr])
    metrics = {"C": C, "class_weight": a.class_weight, "layer": a.layer, "pooling": a.pooling,
               "headline_val_verified_only": eval_block(y[va_ev], pv_ev),
               "secondary_val_all_steps": eval_block(y[va], pv),
               "step_index_baseline_val_verified_auroc":
                   float(roc_auc_score(y[va_ev], pos_clf.predict_proba(pos[va_ev])[:, 1])),
               "n_train_steps": int(tr.sum()), "n_train_trajectories": int(df.loc[tr, "sample_id"].nunique())}

    if "source" in df.columns and df["source"].nunique() > 1:
        per = {}
        for src in sorted(df["source"].unique()):
            m_ = va_ev & (df["source"] == src).to_numpy()
            if len(np.unique(y[m_])) == 2:
                per[src] = eval_block(y[m_], predict(scaler, clf, X[m_]))
        metrics["headline_val_verified_per_source"] = per
        src_oh = pd.get_dummies(df["source"]).to_numpy().astype(float)
        sc_clf = LogisticRegression(max_iter=1000).fit(src_oh[tr], y[tr])
        metrics["source_only_baseline_val_verified_auroc"] = float(
            roc_auc_score(y[va_ev], sc_clf.predict_proba(src_oh[va_ev])[:, 1]))

    # probabilities for every row: OOF (shared problem folds) for train, final probe for val/test
    probs = np.zeros(len(df))
    origin = np.where(tr, "oof", "final").astype(object)
    fold = assign_folds(df[GROUP_COLUMN])
    for k in range(a.folds):
        b_idx = np.where(tr & (fold == k))[0]
        a_idx = np.where(tr & (fold != k))[0]
        if len(b_idx) == 0:
            continue
        if len(np.unique(y[a_idx])) < 2:
            raise ValueError("A CV fold has a single class; use fewer folds or more data.")
        sc, c = fit_probe(X[a_idx], y[a_idx], C, cw)
        probs[b_idx] = predict(sc, c, X[b_idx])
    rest = np.where(~tr)[0]
    probs[rest] = predict(scaler, clf, X[rest])
    tr_ev = tr & ver
    metrics["train_oof_verified_only"] = eval_block(y[tr_ev], probs[tr_ev])

    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler, "C": C, "class_weight": a.class_weight,
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
