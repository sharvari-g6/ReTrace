"""Turn the raw ARS trial log (JSONL) into per-step ARS scores.

ARS (instability) = 1 - (agreeing valid trials / valid trials). Higher = less stable = more
suspicious, the same orientation as the other three signals. Agreement is recomputed here
from the raw generated text, so the equivalence rule can change without regenerating.

    python -m signals.ars.aggregate --trials features/ars_trials.jsonl \
        --output features/ars_scores.csv --diagnostics features/ars_diagnostics.csv
Outputs
  ars_scores.csv       sample_id, step_id, ARS                       (one signal column, for the merge)
  ars_diagnostics.csv  sample_id, step_id, n_trials, n_valid, n_agree, orig_answer, ARS_control, status
Steps with fewer than --min-valid valid trials are NOT written to ars_scores.csv (the matrix
merge then fails loudly instead of silently inventing a value).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from signals.ars.answers import extract_answer, answers_agree


def load_trials(path) -> list[dict]:
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs[(r["sample_id"], r["step_id"])] = r          # last write wins (resume safe)
    return list(recs.values())


def _rate(orig, raws):
    vals = [answers_agree(orig, extract_answer(x)) for x in raws]
    valid = [v for v in vals if v is not None]
    return len(valid), sum(valid)


def step_stats(rec: dict) -> dict:
    orig = extract_answer(rec.get("orig_raw"))
    n_valid, n_agree = _rate(orig, [t["raw"] for t in rec.get("trials", [])]) if orig else (0, 0)
    out = {"sample_id": rec["sample_id"], "step_id": rec["step_id"], "status": rec.get("status", "ok"),
           "n_trials": len(rec.get("trials", [])), "n_valid": n_valid, "n_agree": n_agree,
           "orig_answer": orig, "ARS": (1 - n_agree / n_valid) if n_valid else None,
           "ARS_control": None}
    if rec.get("control") and orig:
        cv, ca = _rate(orig, [t["raw"] for t in rec["control"]])
        out["ARS_control"] = (1 - ca / cv) if cv else None
    return out


def build_scores(records: list[dict], min_valid: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    diag = pd.DataFrame([step_stats(r) for r in records]).sort_values(["sample_id", "step_id"])
    good = diag[(diag["status"] == "ok") & (diag["n_valid"] >= min_valid)]
    return good[["sample_id", "step_id", "ARS"]].reset_index(drop=True), diag.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", default="features/ars_trials.jsonl")
    p.add_argument("--output", default="features/ars_scores.csv")
    p.add_argument("--diagnostics", default="features/ars_diagnostics.csv")
    p.add_argument("--min-valid", type=int, default=2)
    a = p.parse_args()
    scores, diag = build_scores(load_trials(a.trials), a.min_valid)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(a.output, index=False)
    diag.to_csv(a.diagnostics, index=False)
    dropped = len(diag) - len(scores)
    print(f"Wrote {a.output}: {len(scores)} steps ({diag['sample_id'].nunique()} trajectories)")
    print(f"Wrote {a.diagnostics}. Steps without a usable ARS (not in the scores file): {dropped}")
    if dropped:
        print(diag.loc[~diag.index.isin(scores.index) & ((diag['status'] != 'ok') | (diag['n_valid'] < a.min_valid)),
                       "status"].value_counts().to_string())
    print("ARS distribution:", scores["ARS"].round(2).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
