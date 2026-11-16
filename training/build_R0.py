"""Build the GeoReason-inspired reference geometry (R0) and the Deviation signal (Member B's task).

    python -m training.build_R0 --hidden-dir train_features_raw --labels training/labels_split.csv

For every trajectory and step it writes  features/deviation_scores.csv  (sample_id, step_id, Deviation).
Leakage protocol
  * train trajectories are scored OUT-OF-FOLD (shared folds, features/folds.py): the lens U, the
    normalisation and the reference cloud R0 used for a train trajectory never saw its problem;
  * val / test trajectories are scored with the model fit on ALL train trajectories
    (saved as models/R0.npz, the artifact to reuse at inference);
  * labels are used to fit on train trajectories only, never to score (center != oracle).
Step 0 needs no question-end state: missing previous steps are zero-padded in the lens space (paper Eq. 6).
Higher Deviation = further from the cloud of correct transitions = more suspicious.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from features.folds import assign_folds, N_FOLDS
from signals.geo.io import load_trajectories
from signals.geo import geometry as G


def metrics(df, mask):
    d = df[mask]
    if d["label"].nunique() < 2:
        return None
    return {"n": int(len(d)), "positives": int(d["label"].sum()),
            "auroc": float(roc_auc_score(d["label"], d["Deviation"])),
            "pr_auc": float(average_precision_score(d["label"], d["Deviation"]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden-dir", required=True)
    p.add_argument("--labels", default="training/labels_split.csv")
    p.add_argument("--layer", type=int, default=-1, choices=[-8, -4, -1])
    p.add_argument("--pooling", default="mean", choices=["mean", "last_token"])
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--center", default="global", choices=["global", "trace_mean", "oracle"])
    p.add_argument("--cost", default="maha", choices=["euclid", "diag", "maha"])
    p.add_argument("--model-out", default="models/R0.npz")
    p.add_argument("--output", default="features/deviation_scores.csv")
    p.add_argument("--report", default="artifacts/deviation_report.json")
    a = p.parse_args()
    cfg = G.Config(k=a.k, alpha=a.alpha, rho=a.rho, center=a.center, cost=a.cost)
    out_path = a.output
    if a.center == "oracle":
        print("DIAGNOSTIC ONLY: 'oracle' normalisation uses labels at scoring time. Not a valid RFS feature.")
        out_path = a.output.replace(".csv", "_ORACLE.csv")

    trajs = load_trajectories(a.hidden_dir, a.labels, a.layer, a.pooling)
    train = [t for t in trajs if t["split"] == "train"]
    others = [t for t in trajs if t["split"] != "train"]
    if not train:
        raise SystemExit("No train trajectories found.")
    folds = assign_folds([t["problem_id"] for t in train])
    rows = []

    def emit(t, s, origin):
        for i, v in enumerate(s):
            rows.append((t["sample_id"], i, float(v), t["split"], t["source"], int(t["label"][i]),
                         int(t["verified"][i]), origin))

    for k in range(N_FOLDS):
        fit_t = [t for t, f in zip(train, folds) if f != k]
        eval_t = [t for t, f in zip(train, folds) if f == k]
        if not eval_t:
            continue
        assert not ({t["problem_id"] for t in fit_t} & {t["problem_id"] for t in eval_t}), "fold leakage"
        m = G.fit(fit_t, cfg)
        for t in eval_t:
            emit(t, G.score(m, t["H"], t["label"]), "oof")
        print(f"fold {k}: fit on {len(fit_t)} trajectories, scored {len(eval_t)}", flush=True)
    final = G.fit(train, cfg)
    Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
    G.save(final, a.model_out)
    for t in others:
        emit(t, G.score(final, t["H"], t["label"]), "final")

    df = pd.DataFrame(rows, columns=["sample_id", "step_id", "Deviation", "split", "source", "label", "verified", "origin"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df[["sample_id", "step_id", "Deviation"]].sort_values(["sample_id", "step_id"]).to_csv(out_path, index=False)
    ver = df["verified"] == 1
    rep = {"config": vars(cfg), "layer": a.layer, "pooling": a.pooling, "final_model_info": final.info,
           "headline_verified_only": {sp: metrics(df, ver & (df["split"] == sp)) for sp in ["train", "val", "test"]
                                      if sp != "test"},
           "all_steps_secondary": {sp: metrics(df, df["split"] == sp) for sp in ["train", "val"]},
           "per_source_verified_train_oof_plus_val": {
               s: metrics(df, ver & (df["source"] == s) & (df["split"] != "test")) for s in sorted(df["source"].unique())},
           "note": "test split is scored but NOT evaluated here"}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=2))
    print(f"Wrote {out_path}: {len(df)} steps, {df['sample_id'].nunique()} trajectories; model -> {a.model_out}")
    print(json.dumps(rep["headline_verified_only"], indent=2))


if __name__ == "__main__":
    main()
