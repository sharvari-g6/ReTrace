"""Download the raw datasets into data/raw/ in the layout the loaders expect.

Usage (from the repo root):
    python data/download_raw.py                 # everything
    python data/download_raw.py --only prm800k  # one dataset (processbench | halueval | prm800k)
    python data/download_raw.py --force         # re-download even if files exist

Only the files the loaders actually use are fetched:
  ProcessBench : all four splits
  HaluEval     : qa and dialogue (paired files); summarization etc. are skipped
  PRM800K      : phase 2 train/test (large files; the download can take a while)
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"

# expected row counts, used only as a sanity check (a mismatch is a warning, not an error)
PROCESSBENCH_SPLITS = {"gsm8k": 400, "math": 1000, "olympiadbench": 1000, "omnimath": 1000}
HALUEVAL_CONFIGS = {"qa": 10000, "dialogue": 10000}
PRM800K_FILES = {"phase2_train.jsonl": 97782, "phase2_test.jsonl": 2762}

# media.githubusercontent.com serves Git LFS files directly, so git-lfs is not needed
PRM800K_URL = "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/{name}"


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def looks_like_lfs_pointer(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readline().startswith("version https://git-lfs")


def check(path, expected):
    """Print the row count of a downloaded file; return True if it matches."""
    n = count_lines(path)
    ok = n == expected
    status = "OK" if ok else f"WARNING: expected {expected} rows"
    print(f"  {path.relative_to(REPO_ROOT)}: {n} rows  [{status}]")
    return ok


def download_processbench(force):
    print("\n== ProcessBench ==")
    out_dir = RAW / "processbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {s: out_dir / f"processbench_{s}.jsonl" for s in PROCESSBENCH_SPLITS}

    todo = [s for s, p in targets.items() if force or not p.exists()]
    if todo:
        from datasets import load_dataset  # imported here so --only prm800k needs no `datasets`
        ds = load_dataset("Qwen/ProcessBench")
        missing = [s for s in todo if s not in ds]
        if missing:
            raise RuntimeError(f"ProcessBench is missing expected splits: {missing}")
        for s in todo:
            tmp = targets[s].with_suffix(".jsonl.tmp")
            ds[s].to_json(str(tmp), orient="records", lines=True)
            os.replace(tmp, targets[s])
    else:
        print("  all files already present (use --force to re-download)")

    results = [check(targets[s], n) for s, n in PROCESSBENCH_SPLITS.items()]
    return all(results)


def download_halueval(force):
    print("\n== HaluEval ==")
    out_dir = RAW / "halueval"
    out_dir.mkdir(parents=True, exist_ok=True)
    # the HF dataset has a single split called "data" -> files qa_data.jsonl, dialogue_data.jsonl
    targets = {c: out_dir / f"{c}_data.jsonl" for c in HALUEVAL_CONFIGS}

    todo = [c for c, p in targets.items() if force or not p.exists()]
    if todo:
        from datasets import load_dataset
        for cfg in todo:
            ds = load_dataset("pminervini/HaluEval", cfg)
            if "data" not in ds:
                raise RuntimeError(f"HaluEval/{cfg}: expected a 'data' split, got {list(ds.keys())}")
            tmp = targets[cfg].with_suffix(".jsonl.tmp")
            ds["data"].to_json(str(tmp), orient="records", lines=True)
            os.replace(tmp, targets[cfg])
    else:
        print("  all files already present (use --force to re-download)")

    results = [check(targets[c], n) for c, n in HALUEVAL_CONFIGS.items()]
    return all(results)


def download_file(url, dest):
    import requests
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                size = f" / {total / 1e6:.0f} MB" if total else ""
                print(f"\r  {dest.name}: {done / 1e6:.0f} MB{size}", end="", flush=True)
    print()
    if looks_like_lfs_pointer(tmp):
        tmp.unlink()
        raise RuntimeError("received a Git LFS pointer instead of the data file")
    os.replace(tmp, dest)


def download_prm800k(force):
    print("\n== PRM800K (phase 2; large files) ==")
    out_dir = RAW / "prm800k"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in PRM800K_FILES:
        dest = out_dir / name
        if dest.exists() and not force and not looks_like_lfs_pointer(dest):
            print(f"  {name}: already present (use --force to re-download)")
            continue
        try:
            download_file(PRM800K_URL.format(name=name), dest)
        except Exception as e:
            print(
                f"\n  Could not download {name}: {type(e).__name__}: {e}\n"
                "  Manual fallback (needs git and git-lfs):\n"
                "    git lfs install\n"
                "    git clone https://github.com/openai/prm800k.git\n"
                "    cd prm800k && git lfs pull\n"
                f"    then copy prm800k/data/{name} into data/raw/prm800k/"
            )
            raise

    results = [check(out_dir / name, n) for name, n in PRM800K_FILES.items()]
    return all(results)


DOWNLOADERS = {
    "processbench": download_processbench,
    "halueval": download_halueval,
    "prm800k": download_prm800k,
}


def main():
    parser = argparse.ArgumentParser(description="Download raw datasets into data/raw/")
    parser.add_argument("--only", choices=list(DOWNLOADERS), help="download a single dataset")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    names = [args.only] if args.only else list(DOWNLOADERS)
    failed, mismatched = [], []
    for name in names:
        try:
            if not DOWNLOADERS[name](args.force):
                mismatched.append(name)
        except Exception as e:
            print(f"\n[{name}] FAILED: {type(e).__name__}: {e}")
            failed.append(name)

    print("\n== Summary ==")
    print("Downloaded OK:", [n for n in names if n not in failed and n not in mismatched] or "none")
    if mismatched:
        print("Row counts differ from expected for:", mismatched,
              "- the dataset version may have changed; check the fingerprint after make_splits.")
    if failed:
        print("Failed:", failed)
        sys.exit(1)
    print("\nNext:")
    print("  python -m data.loaders.processbench")
    print("  python -m data.loaders.prm800k")
    print("  python -m data.loaders.halueval")
    print("  python -m data.make_splits   # then compare the fingerprint with data/splits/split_manifest.json")


if __name__ == "__main__":
    main()