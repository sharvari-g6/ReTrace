"""Print the real fields of the locally downloaded datasets (read-only)."""
import json
from pathlib import Path

RAW = Path("data/raw")

def preview(path, n=1, maxlen=400):
    print(f"\n===== {path} =====")
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        if first.startswith("version https://git-lfs"):
            print("!! Git LFS pointer, not the real file. Run `git lfs pull`.")
            return
        rows = [json.loads(first)]
        for _ in range(n - 1):
            line = f.readline()
            if line:
                rows.append(json.loads(line))
    for i, row in enumerate(rows):
        print(f"--- example {i}: keys = {list(row.keys())}")
        for k, v in row.items():
            s = repr(v)
            print(f"  {k}: {s[:maxlen]}{'...' if len(s) > maxlen else ''}")

def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

for p in sorted(RAW.rglob("*.jsonl")):
    preview(p)
    print(f"  rows: {count_lines(p)}")