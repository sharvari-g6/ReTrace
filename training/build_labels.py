"""Build the step-level label table from the hidden-state shards.

Labels, split, problem_id and meta.verified_steps are read from the records themselves and
cross-checked against data/splits/problem_splits.csv (Nitish's manifest).

    python -m training.build_labels --hidden-dir train_features_raw \
        --output training/labels_split.csv [--sample-list sampled_ids.csv] [--drop-after-error]

* Records with step_labels = None (HaluEval, answer-level only) are NOT written to the step
  table. They go to --answer-output (sample_id, problem_id, split, source, answer_label) for
  the answer-level / out-of-domain evaluation later.
* --sample-list : CSV/txt with a `sample_id` column (or one id per line). Only those
  trajectories are kept (use it for the agreed sampled pilot set).
* verified = 1 for step_id < meta.verified_steps (headline evaluation uses these only).
* --drop-after-error : masked-training ablation (drop unverified steps = zero weight).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from features.validate_schema import validate_matrix


def read_shards(hidden_dir):
    import torch
    for path in sorted(Path(hidden_dir).rglob("*.pt")):   # searches subfolders (same file names allowed per dataset)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        for rec in obj["records"]:
            yield rec


def read_sample_list(path):
    df = pd.read_csv(path)
    col = "sample_id" if "sample_id" in df.columns else df.columns[0]
    return set(df[col].astype(str))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden-dir", required=True)
    p.add_argument("--output", default="training/labels_split.csv")
    p.add_argument("--answer-output", default="training/answer_level.csv")
    p.add_argument("--manifest", default="data/splits/problem_splits.csv")
    p.add_argument("--sample-list", default=None)
    p.add_argument("--drop-after-error", action="store_true")
    a = p.parse_args()

    wanted = read_sample_list(a.sample_list) if a.sample_list else None
    rows, answers, seen = [], [], set()
    for r in read_shards(a.hidden_dir):
        sid = r["sample_id"]
        if wanted is not None and sid not in wanted:
            continue
        if sid in seen:
            raise ValueError(f"{sid} appears in more than one shard.")
        seen.add(sid)
        if r["step_labels"] is None:                       # answer-level only (HaluEval)
            answers.append((sid, r["problem_id"], r["split"], r["source"], r["answer_label"]))
            continue
        fe, nv = r["first_error"], r["meta"]["verified_steps"]
        for i, lab in enumerate(r["step_labels"]):
            rows.append((sid, i, r["problem_id"], r["split"], int(lab), int(i < nv),
                         fe is not None and i > fe, r["source"], r["meta"].get("subset")))
    if wanted is not None and (wanted - seen):
        print(f"WARNING: {len(wanted - seen)} ids in --sample-list were not found in the shards.")

    df = pd.DataFrame(rows, columns=["sample_id", "step_id", "problem_id", "split", "label",
                                     "verified", "after_error", "source", "subset"])
    man = pd.read_csv(a.manifest).set_index("problem_id")["split"] if Path(a.manifest).exists() else None
    if man is not None:
        wrong = (df["problem_id"].map(man) != df["split"]).sum()
        if wrong:
            raise ValueError(f"{wrong} rows disagree with {a.manifest}. Re-extract or re-split.")
    if a.drop_after_error:
        df = df[~df["after_error"]]
    df = validate_matrix(df, require_signals=False, require_all_splits=False)
    df.to_csv(a.output, index=False)
    print(f"Wrote {a.output}: {df['sample_id'].nunique()} trajectories, {len(df)} steps")
    v = df[df["verified"] == 1]
    print("Verified rows (headline evaluation):")
    print(v.groupby(["source", "split"])["label"].agg(rows="count", positives="sum").to_string())

    if answers:
        ans = pd.DataFrame(answers, columns=["sample_id", "problem_id", "split", "source", "answer_label"])
        if man is not None and (ans["problem_id"].map(man) != ans["split"]).any():
            raise ValueError("Answer-level rows disagree with the split manifest.")
        ans.to_csv(a.answer_output, index=False)
        print(f"Wrote {a.answer_output}: {len(ans)} answer-level samples "
              f"(kept out of probe/RFS training), label balance {ans['answer_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
