"""Problem-group train/val/test split (policy agreed with Member A).

Policy:
  * Global grouping by normalized-problem hash (see data/unified.py) across
    processbench, prm800k and halueval.
  * A problem appearing in several sources is ONE group -> exactly one split.
  * Split is done BEFORE any subsampling, stratified by source composition.
  * ProcessBench is part of the grouped data: it is NOT an untouched benchmark.

Outputs:
  data/splits/split_manifest.json   (shared: seed, policy, overlap, counts, fingerprint)
  data/splits/problem_splits.csv    (shared: problem_id, split, sources)
  data/processed/sample_splits.csv  (git-ignored, regenerated: sample_id, source, problem_id, split)
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

PROCESSED = Path("data/processed")
SPLIT_DIR = Path("data/splits")
SOURCES = {
    "processbench": "processbench.jsonl",
    "prm800k": "prm800k.jsonl",
    "halueval": "halueval.jsonl",
}
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15


def read_light(path):
    """Yield (sample_id, source, problem_id, is_positive) without keeping step text."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["step_labels"] is not None:
                positive = r["first_error"] is not None      # trajectory contains a step error
            else:
                positive = r["answer_label"] == 1             # HaluEval: hallucinated response
            yield r["sample_id"], r["source"], r["problem_id"], positive


def main():
    samples = []
    for src, fname in SOURCES.items():
        samples.extend(read_light(PROCESSED / fname))
    assert len({s[0] for s in samples}) == len(samples), "duplicate sample_id across sources"

    # ---- groups ----
    group_sources = defaultdict(set)
    for _, src, pid, _ in samples:
        group_sources[pid].add(src)
    pids = sorted(group_sources)                                  # deterministic order
    strata = ["+".join(sorted(group_sources[p])) for p in pids]

    # ---- 70 / 15 / 15 split of GROUPS ----
    train_p, temp_p, _, temp_strata = train_test_split(
        pids, strata, test_size=VAL_FRAC + TEST_FRAC,
        random_state=SEED, stratify=strata)
    val_p, test_p = train_test_split(
        temp_p, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        random_state=SEED, stratify=temp_strata)
    split_of = {p: "train" for p in train_p}
    split_of.update({p: "val" for p in val_p})
    split_of.update({p: "test" for p in test_p})

    # ---- invariants ----
    assert set(split_of) == set(group_sources), "some group has no split"
    assert len(train_p) + len(val_p) + len(test_p) == len(pids)
    assert not (set(train_p) & set(val_p) or set(train_p) & set(test_p) or set(val_p) & set(test_p))

    # ---- overlap report ----
    overlap = Counter("+".join(sorted(s)) for s in group_sources.values())
    print("Groups by source composition:", dict(overlap))
    shared = overlap.get("prm800k+processbench", 0)
    print(f"ProcessBench problems also present in PRM800K: {shared}")

    # ---- counts per split x source ----
    stats = defaultdict(lambda: {"problems": set(), "trajectories": 0, "positive": 0})
    for sid, src, pid, pos in samples:
        c = stats[(split_of[pid], src)]
        c["problems"].add(pid)
        c["trajectories"] += 1
        c["positive"] += pos

    counts = {}
    print(f"\n{'split':6s} {'source':13s} {'problems':>9s} {'trajectories':>13s} {'positive':>9s}")
    for split in ("train", "val", "test"):
        counts[split] = {}
        for src in SOURCES:
            c = stats[(split, src)]
            counts[split][src] = {"problems": len(c["problems"]),
                                  "trajectories": c["trajectories"],
                                  "positive": c["positive"]}
            print(f"{split:6s} {src:13s} {len(c['problems']):9d} {c['trajectories']:13d} {c['positive']:9d}")
        n_groups = sum(1 for p in split_of if split_of[p] == split)
        n_traj = sum(counts[split][s]["trajectories"] for s in SOURCES)
        counts[split]["TOTAL"] = {"unique_problem_groups": n_groups, "trajectories": n_traj}
        print(f"{split:6s} {'TOTAL':13s} {n_groups:9d} {n_traj:13d}")
        print()
    print("(a problem shared by two sources is counted once in TOTAL but once per source above)")

    # ---- write files ----
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_DIR / "problem_splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem_id", "split", "sources"])
        for p in pids:
            w.writerow([p, split_of[p], "+".join(sorted(group_sources[p]))])

    with open(PROCESSED / "sample_splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "source", "problem_id", "split"])
        for sid, src, pid, _ in sorted(samples):
            w.writerow([sid, src, pid, split_of[pid]])

    # fingerprint lets teammates check their processed data matches before using the manifest
    fingerprint = hashlib.sha1(
        "\n".join(f"{sid}:{pid}" for sid, _, pid, _ in sorted(samples)).encode("utf-8")
    ).hexdigest()

    manifest = {
        "seed": SEED,
        "ratios": {"train": 1 - VAL_FRAC - TEST_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "policy": (
            "Global grouping by normalized-problem hash (NFKC, lowercase, whitespace collapsed, "
            "sha1[:16]) across processbench, prm800k and halueval. A problem present in several "
            "sources is one group assigned to exactly one split. Split by group, stratified by "
            "source composition, before any subsampling. ProcessBench is part of the grouped "
            "data and is NOT an untouched benchmark. Normalization statistics, RFS weights and "
            "thresholds must be fitted without using test data."
        ),
        "group_counts_by_source_composition": dict(overlap),
        "counts": counts,
        "samples_fingerprint_sha1": fingerprint,
        "n_samples": len(samples),
        "n_problem_groups": len(pids),
    }
    with open(SPLIT_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {SPLIT_DIR / 'split_manifest.json'}, {SPLIT_DIR / 'problem_splits.csv'}, "
          f"{PROCESSED / 'sample_splits.csv'}")
    print("Fingerprint:", fingerprint)


if __name__ == "__main__":
    main()