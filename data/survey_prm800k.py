import json
from collections import Counter
from pathlib import Path

RAW = Path("data/raw/prm800k")


def survey(path):
    print(f"\n===== {path.name} =====")
    q_keys, l_keys = Counter(), Counter()
    finish, n_comp, ratings = Counter(), Counter(), Counter()
    n = 0
    n_steps, problems = [], set()
    steps_comp_none = steps_chosen_none = steps_human = 0
    both_none = comp_none_but_human = comp_none_no_human = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            q_keys.update(r["question"].keys())
            l_keys.update(r["label"].keys())
            finish[r["label"].get("finish_reason")] += 1
            problems.add(r["question"]["problem"])
            steps = r["label"]["steps"] or []
            n_steps.append(len(steps))
            for s in steps:
                comps = s.get("completions")
                chosen = s.get("chosen_completion")
                human = s.get("human_completion")
                if comps is None:
                    steps_comp_none += 1
                    if human is not None:
                        comp_none_but_human += 1
                    else:
                        comp_none_no_human += 1
                else:
                    n_comp[len(comps)] += 1
                    for c in comps:
                        ratings[c.get("rating")] += 1
                if chosen is None:
                    steps_chosen_none += 1
                if human is not None:
                    steps_human += 1
    print("rows:", n, "| unique problems:", len(problems))
    print("question keys:", dict(q_keys))
    print("label keys:", dict(l_keys))
    print("finish_reason:", dict(finish))
    print("completions per step (non-null):", dict(n_comp))
    print("ratings:", dict(ratings))
    print("steps with completions=None:", steps_comp_none,
          "| of which human_completion present:", comp_none_but_human,
          "| absent:", comp_none_no_human)
    print("steps with chosen_completion=None:", steps_chosen_none,
          "| with human_completion:", steps_human)
    if n_steps:
        print(f"steps per row: min={min(n_steps)} max={max(n_steps)} "
              f"mean={sum(n_steps) / len(n_steps):.1f}")


def show_example(path, row_idx=0):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == row_idx:
                r = json.loads(line)
                break
    print(f"\n--- example {row_idx} from {path.name} ---")
    print("problem:", r["question"]["problem"][:150])
    print("finish_reason:", r["label"].get("finish_reason"))
    for j, s in enumerate(r["label"]["steps"] or []):
        human = s.get("human_completion")
        print(f" step {j}: chosen={s.get('chosen_completion')} "
              f"human={'yes' if human else 'no'}")
        if human:
            print(f"    human: {str(human)[:90]!r}")
        for k, c in enumerate(s.get("completions") or []):
            print(f"    [{k}] rating={c.get('rating')} | {str(c.get('text'))[:90]!r}")


def show_example_with_error(path):
    """First example whose steps contain a rating of -1, to see how errors appear."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            for s in r["label"]["steps"] or []:
                if any(c.get("rating") == -1 for c in (s.get("completions") or [])):
                    show_example(path, i)
                    return


for name in ["phase1_train.jsonl", "phase2_train.jsonl"]:
    survey(RAW / name)
show_example(RAW / "phase1_train.jsonl", 0)
show_example(RAW / "phase2_train.jsonl", 0)
print("\n########## example containing a -1 rating (phase2) ##########")
show_example_with_error(RAW / "phase2_train.jsonl")