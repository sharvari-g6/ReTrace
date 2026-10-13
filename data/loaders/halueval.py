"""HaluEval (paired qa + dialogue files) -> unified format. Answer-level labels only.

Each pair yields two single-step samples: the right response (0) and the hallucinated one (1).
step_labels/first_error are None: HaluEval has no step annotations.
Both samples of a pair share the same prompt text, so they share a problem_id
and stay in the same split.
Output: data/processed/halueval.jsonl  (git-ignored)
"""
import json
from collections import Counter
from pathlib import Path

from data.unified import make_record, CORRECT, ERROR

RAW_DIR = Path("data/raw/halueval")
OUT_PATH = Path("data/processed/halueval.jsonl")

TASKS = {
    "qa": {"file": "qa_data.jsonl", "prompt": "question",
           "right": "right_answer", "bad": "hallucinated_answer"},
    "dialogue": {"file": "dialogue_data.jsonl", "prompt": "dialogue_history",
                 "right": "right_response", "bad": "hallucinated_response"},
}


def load_halueval(raw_dir=RAW_DIR):
    records, counter, skipped = [], 0, Counter()
    for task, cfg in TASKS.items():
        with open(Path(raw_dir) / cfg["file"], "r", encoding="utf-8") as f:
            for pair_idx, line in enumerate(f):
                row = json.loads(line)
                prompt = row[cfg["prompt"]].strip()
                right, bad = row[cfg["right"]].strip(), row[cfg["bad"]].strip()
                if not (prompt and right and bad):
                    skipped[task] += 1
                    continue
                for text, label in ((right, CORRECT), (bad, ERROR)):
                    counter += 1
                    records.append(make_record(
                        sample_id=f"HE_{counter:05d}",
                        source="halueval",
                        question=prompt,
                        steps=[text],
                        step_labels=None,
                        first_error=None,
                        answer=text,
                        answer_label=label,
                        meta={"task": task, "pair_idx": pair_idx,
                              "knowledge": row.get("knowledge")},
                    ))
    return records, skipped


def summarize(records, skipped):
    print(f"Total samples: {len(records)} | skipped pairs: {dict(skipped)}")
    print("Per task:", dict(Counter(r["meta"]["task"] for r in records)))
    print("Answer-label balance:", dict(Counter(r["answer_label"] for r in records)))
    print("Unique problems:", len({r["problem_id"] for r in records}))


if __name__ == "__main__":
    recs, skipped = load_halueval()
    summarize(recs, skipped)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_PATH}")