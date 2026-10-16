"""Day-4-style sanity report for ARS: do perturbations sometimes preserve and sometimes change
the answer, and does ARS separate error steps from correct steps (verified rows only)?

    python -m signals.ars.report --scores features/ars_scores.csv \
        --diagnostics features/ars_diagnostics.csv --labels training/labels_split.csv
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", default="features/ars_scores.csv")
    p.add_argument("--diagnostics", default="features/ars_diagnostics.csv")
    p.add_argument("--labels", default="training/labels_split.csv")
    p.add_argument("--output", default="artifacts/ars_report.json")
    a = p.parse_args()
    s = pd.read_csv(a.scores)
    d = pd.read_csv(a.diagnostics)
    lab = pd.read_csv(a.labels)
    rep = {"n_steps_scored": int(len(s)), "n_steps_attempted": int(len(d))}
    rep["fraction_all_agree"] = float((s["ARS"] == 0).mean())
    rep["fraction_all_disagree"] = float((s["ARS"] == 1).mean())
    rep["fraction_mixed"] = float(((s["ARS"] > 0) & (s["ARS"] < 1)).mean())
    rep["status_counts"] = d["status"].value_counts().to_dict()
    m = s.merge(lab, on=["sample_id", "step_id"], how="inner")
    rep["n_joined_with_labels"] = int(len(m))
    ver = m[m["verified"] == 1] if "verified" in m else m
    for sp in ["train", "val", "test"]:
        x = ver[ver["split"] == sp] if "split" in ver else ver
        if len(x) and x["label"].nunique() == 2:
            rep[f"auroc_verified_{sp}"] = float(roc_auc_score(x["label"], x["ARS"]))
            rep[f"mean_ARS_label0_{sp}"] = float(x.loc[x["label"] == 0, "ARS"].mean())
            rep[f"mean_ARS_label1_{sp}"] = float(x.loc[x["label"] == 1, "ARS"].mean())
    if "ARS_control" in d and d["ARS_control"].notna().any():
        c = d.dropna(subset=["ARS_control", "ARS"])
        rep["mean_ARS_noise"] = float(c["ARS"].mean())
        rep["mean_ARS_control_sampling_only"] = float(c["ARS_control"].mean())
        rep["note"] = ("If the two means are close, the latent noise adds little beyond sampling variability; "
                       "raise --noise-scale or use --noise-mode relative.")
    print(json.dumps(rep, indent=2))
    import os
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    open(a.output, "w").write(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
