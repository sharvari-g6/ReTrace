"""PRM800K (phase 2) -> unified format.

PRM800K ratings: 1 = good, 0 = neutral, -1 = bad.
Phase 2: a labeler walks a pre-generated solution and stops at the first -1 step.
We rebuild the FULL pre-generated solution; labels:
  steps before the error -> 0, error step -> 1,
  steps after the error  -> AFTER_ERROR_LABEL (team convention, not annotated).
Output: data/processed/prm800k.jsonl  (git-ignored, streamed to disk)
"""
import json
from collections import Counter
from pathlib import Path

from data.unified import make_record, CORRECT, ERROR, AFTER_ERROR_LABEL

RAW_DIR = Path("data/raw/prm800k")
OUT_PATH = Path("data/processed/prm800k.jsonl")
FILES = ["phase2_train.jsonl", "phase2_test.jsonl"]  # fixed order => stable IDs
KEEP_FINISH = {"found_error", "solution"}            # give_up / bad_problem dropped
NEUTRAL_IS_ERROR = False                              # rating 0 -> not an error (confirm with Member A)


def _norm(t):
    return " ".join(str(t).split())


def _is_error(rating):
    return rating == -1 or (NEUTRAL_IS_ERROR and rating == 0)


def convert_row(row):
    """Return (fields, None) on success or (None, skip_reason)."""
    label = row["label"]
    finish = label.get("finish_reason")
    if finish not in KEEP_FINISH:
        return None, f"finish_{finish}"

    labeled = label.get("steps") or []
    pre = row["question"].get("pre_generated_steps")
    if not labeled:
        return None, "no_labeled_steps"
    if not isinstance(pre, list) or not pre:
        return None, "no_pre_generated_steps"
    if len(labeled) > len(pre):
        return None, "labeled_longer_than_pregen"

    ratings = []
    for j, s in enumerate(labeled):
        comps = s.get("completions") or []
        chosen = s.get("chosen_completion")
        target = _norm(pre[j])
        if chosen is not None and chosen < len(comps):
            cand = comps[chosen]
        else:  # the stopping step: pick the completion that is the solution's own text
            cand = next((c for c in comps if _norm(c.get("text", "")) == target), None)
        if cand is None or _norm(cand.get("text", "")) != target:
            return None, "text_mismatch_with_pregen"
        if cand.get("rating") is None:
            return None, "rating_none"
        ratings.append(cand["rating"])

    errs = [j for j, r in enumerate(ratings) if _is_error(r)]
    n = len(pre)
    if finish == "found_error":
        if errs != [len(labeled) - 1]:
            return None, "found_error_without_single_final_error"
        first_error = len(labeled) - 1
        step_labels = [CORRECT] * first_error + [ERROR] + [AFTER_ERROR_LABEL] * (n - first_error - 1)
        verified = first_error + 1
    else:  # solution
        if errs:
            return None, "solution_with_error_rating"
        if len(labeled) != n:
            return None, "solution_partially_labeled"
        first_error = None
        step_labels = [CORRECT] * n
        verified = n

    q = row["question"]
    gt = q.get("ground_truth_answer")
    ans = q.get("pre_generated_answer")
    return {
        "question": q["problem"],
        "steps": pre,
        "step_labels": step_labels,
        "first_error": first_error,
        "answer": ans,
        "meta": {
            "finish_reason": finish,
            "verified_steps": verified,
            "ground_truth_answer": gt,
            # weak signal only: plain string match is noisy for LaTeX answers
            "answer_exact_match": _norm(ans) == _norm(gt) if ans is not None and gt is not None else None,
        },
    }, None


def build(raw_dir=RAW_DIR, out_path=OUT_PATH):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skips, kept_finish = Counter(), Counter()
    kept = raw_rows = 0
    n_steps_total = n_err = err_is_last = 0
    problems = set()
    counter = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for fname in FILES:
            with open(Path(raw_dir) / fname, "r", encoding="utf-8") as f:
                for line in f:
                    counter += 1
                    raw_rows += 1
                    fields, why = convert_row(json.loads(line))
                    if why:
                        skips[why] += 1
                        continue
                    fields["meta"]["file"] = fname
                    rec = make_record(sample_id=f"PR_{counter:06d}", source="prm800k",
                                      answer_label=None, **fields)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    kept_finish[rec["meta"]["finish_reason"]] += 1
                    problems.add(rec["problem_id"])
                    n_steps_total += len(rec["steps"])
                    if rec["first_error"] is not None:
                        n_err += 1
                        err_is_last += rec["first_error"] == len(rec["steps"]) - 1

    print(f"Raw rows: {raw_rows} | kept: {kept} | unique problems kept: {len(problems)}")
    print("Kept by finish_reason:", dict(kept_finish))
    print("Skipped:", dict(skips))
    print(f"Mean steps/trajectory: {n_steps_total / max(kept, 1):.2f}")
    print(f"With a step error: {n_err} ({100 * n_err / max(kept, 1):.1f}%)")
    if n_err:
        print(f"First error is the LAST step in {100 * err_is_last / n_err:.1f}% of error trajectories")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    build()