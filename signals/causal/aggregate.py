"""Raw patching trials -> per-step Causal score.

  Causal_t = mean_r clip( log p_clean(a*) - log p_patched_{t,r}(a*), -clip, +clip )   (nats)
Positive = patching step t weakens the model's own answer (step t causally supports it).
IMPORTANT: whether a larger value means "more suspicious" is NOT assumed. The logistic RFS learns the
sign of the weight; the report prints the AUROC so you can see which way the signal points.

    python -m signals.causal.aggregate --trials features/causal_trials.jsonl \
        --output features/causal_scores.csv --diagnostics features/causal_diagnostics.csv
    python -m signals.causal.aggregate ... --labels training/labels_split.csv     # adds the sanity report
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load(path):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[(r["sample_id"], r["step_id"])] = r
    return list(recs.values())


def build(records, clip=20.0, min_trials=1):
    rows = []
    for r in records:
        row = {"sample_id": r["sample_id"], "step_id": r["step_id"], "status": r.get("status", "ok"),
               "Causal": None, "effect_per_token": None, "n_trials": 0, "logp_clean": r.get("logp_clean")}
        if row["status"] == "ok" and r.get("logp_patch"):
            eff = np.clip(r["logp_clean"] - np.asarray(r["logp_patch"], float), -clip, clip)
            if len(eff) >= min_trials and np.isfinite(eff).all():
                row["Causal"] = float(eff.mean())
                row["effect_per_token"] = row["Causal"] / max(r.get("n_ans_tokens", 1), 1)
                row["n_trials"] = int(len(eff))
        rows.append(row)
    diag = pd.DataFrame(rows).sort_values(["sample_id", "step_id"]).reset_index(drop=True)
    scores = diag.dropna(subset=["Causal"])[["sample_id", "step_id", "Causal"]].reset_index(drop=True)
    return scores, diag


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", default="features/causal_trials.jsonl")
    p.add_argument("--output", default="features/causal_scores.csv")
    p.add_argument("--diagnostics", default="features/causal_diagnostics.csv")
    p.add_argument("--clip", type=float, default=20.0, help="fixed cap (nats) on a single trial's effect")
    p.add_argument("--labels", default=None)
    a = p.parse_args()
    scores, diag = build(load(a.trials), a.clip)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(a.output, index=False)
    diag.to_csv(a.diagnostics, index=False)
    print(f"Wrote {a.output}: {len(scores)} of {len(diag)} steps scored; status counts {diag['status'].value_counts().to_dict()}")
    print("Causal quantiles:", scores["Causal"].quantile([0, .1, .5, .9, 1]).round(2).to_dict())
    if a.labels:
        from sklearn.metrics import roc_auc_score
        lab = pd.read_csv(a.labels)
        m = scores.merge(lab, on=["sample_id", "step_id"])
        v = m[m["verified"] == 1] if "verified" in m else m
        rep = {}
        for sp in ["train", "val"]:
            x = v[v["split"] == sp]
            if len(x) and x["label"].nunique() == 2:
                rep[sp] = {"n": int(len(x)), "auroc_larger_means_error": float(roc_auc_score(x["label"], x["Causal"])),
                           "mean_label0": float(x.loc[x.label == 0, "Causal"].mean()),
                           "mean_label1": float(x.loc[x.label == 1, "Causal"].mean())}
        print(json.dumps(rep, indent=2))
        print("AUROC < 0.5 means the signal points the other way (larger = LESS likely an error); that is fine, "
              "the RFS weight will be negative.")


if __name__ == "__main__":
    main()
