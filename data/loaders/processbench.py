"""ProcessBench -> unified format.

ProcessBench label: index (0-based) of the earliest erroneous step, or -1 if all steps are correct.
Output: data/processed/processbench.jsonl  (git-ignored)
"""
import json
from collections import Counter
from pathlib import Path

from data.unified import make_record, CORRECT, ERROR, AFTER_ERROR_LABEL

RAW_DIR = Path("data/raw/processbench")
OUT_PATH = Path("data/processed/processbench.jsonl")
SPLITS = ["gsm8k", "math", "olympiadbench", "omnimath"]  # fixed order => stable IDs


def load_processbench(raw_dir=RAW_DIR):
    records = []
    counter = 0
    for split in SPLITS:
        path = Path(raw_dir) / f"processbench_{split}.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                counter += 1

                steps = row["steps"]
                pb_label = row["label"]
                assert pb_label == -1 or 0 <= pb_label < len(steps), (
                    f"bad label {pb_label} for {row['id']} with {len(steps)} steps"
                )

                if pb_label == -1:
                    first_error = None
                    step_labels = [CORRECT] * len(steps)
                else:
                    first_error = pb_label
                    step_labels = (
                        [CORRECT] * first_error
                        + [ERROR]
                        + [AFTER_ERROR_LABEL] * (len(steps) - first_error - 1)
                    )

                records.append(make_record(
                    sample_id=f"PB_{counter:05d}",
                    source="processbench",
                    question=row["problem"],
                    steps=steps,
                    step_labels=step_labels,
                    first_error=first_error,
                    answer=None,  # ProcessBench provides no answer text
                    answer_label=CORRECT if row["final_answer_correct"] else ERROR,
                    meta={
                        "original_id": row["id"],
                        "subset": split,
                        "generator": row["generator"],
                        # labels up to and including first_error are annotation-backed;
                        # later steps are labelled 1 by team convention only
                        "verified_steps": len(steps) if first_error is None else first_error + 1,
                    },
                ))
    return records


def summarize(records):
    print(f"Total trajectories: {len(records)}")
    print("Per subset:", dict(Counter(r["meta"]["subset"] for r in records)))
    n_err = sum(r["first_error"] is not None for r in records)
    print(f"With a step error: {n_err} ({100 * n_err / len(records):.1f}%)")
    print(f"Mean steps/trajectory: {sum(len(r['steps']) for r in records) / len(records):.2f}")
    flat = [l for r in records for l in r["step_labels"]]
    print(f"Step-level label balance: {Counter(flat)}")
    cross = Counter((r["first_error"] is not None, r["answer_label"]) for r in records)
    print("(has_step_error, answer_label) counts:", dict(cross))


if __name__ == "__main__":
    recs = load_processbench()
    summarize(recs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_PATH}")