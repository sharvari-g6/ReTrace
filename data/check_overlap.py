"""Report unique problems per source and ProcessBench <-> PRM800K overlap (read-only)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from data.unified import problem_hash

RAW = Path("data/raw")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


pb_sets, pb_rows = defaultdict(set), Counter()
for p in sorted((RAW / "processbench").glob("processbench_*.jsonl")):
    subset = p.stem.replace("processbench_", "")
    for row in read_jsonl(p):
        pb_sets[subset].add(problem_hash(row["problem"]))
        pb_rows[subset] += 1

prm_sets, prm_rows = defaultdict(set), Counter()
for name in ["phase1_train", "phase1_test", "phase2_train", "phase2_test"]:
    for row in read_jsonl(RAW / "prm800k" / f"{name}.jsonl"):
        prm_sets[name].add(problem_hash(row["question"]["problem"]))
        prm_rows[name] += 1

print("\n== Trajectories vs unique problems ==")
for tag, sets, rows in [("ProcessBench", pb_sets, pb_rows), ("PRM800K", prm_sets, prm_rows)]:
    for k in sets:
        print(f"{tag:13s} {k:16s} rows={rows[k]:6d}  unique_problems={len(sets[k]):6d}")

pb_all = set().union(*pb_sets.values())
prm_all = set().union(*prm_sets.values())
print(f"\nProcessBench total unique problems: {len(pb_all)}")
print(f"PRM800K total unique problems:      {len(prm_all)}")
print(f"Overlap (ProcessBench & PRM800K):   {len(pb_all & prm_all)}")

print("\n== Overlap by ProcessBench subset x PRM800K file ==")
header = "".join(f"{n:>14s}" for n in prm_sets)
print(f"{'':16s}{header}")
for sub, s in pb_sets.items():
    cells = "".join(f"{len(s & prm_sets[n]):14d}" for n in prm_sets)
    print(f"{sub:16s}{cells}")

print("\n== Overlap between PRM800K train and test files (leakage inside PRM800K) ==")
for a, b in [("phase1_train", "phase1_test"), ("phase2_train", "phase2_test"),
             ("phase1_train", "phase2_test"), ("phase2_train", "phase1_test")]:
    print(f"{a} & {b}: {len(prm_sets[a] & prm_sets[b])}")

print("\n== ProcessBench subsets overlapping each other ==")
names = list(pb_sets)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"{names[i]} & {names[j]}: {len(pb_sets[names[i]] & pb_sets[names[j]])}")