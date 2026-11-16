# Day 3 (Member A): run guide

Run everything from the repo root, with the venv active. Shards go in `train_features_raw/`, one subfolder per dataset (git-ignored):
    train_features_raw/processbench/shard_*.pt
    train_features_raw/prm800k/shard_*.pt
    train_features_raw/halueval/shard_*.pt
The code searches subfolders recursively, so identical file names in different datasets are fine.
Put the pilot set in its own folder, `train_features_pilot/`, NOT inside train_features_raw/.

## 1. Build the label table
    python -m training.build_labels --hidden-dir train_features_raw --output training/labels_split.csv
Optional: `--sample-list sampled_ids.csv` (the agreed pilot set), `--drop-after-error` (masked ablation).
Writes training/labels_split.csv (step rows, with `verified` and `source`) and
training/answer_level.csv (HaluEval, kept OUT of probe/RFS training).

## 2. Train the step probe
    python -m training.train_step_probe --matrix training/labels_split.csv \
        --hidden-dir train_features_raw --layer -1 --pooling last_token
Writes artifacts/step_probe.joblib, artifacts/step_probe_validation_metrics.json,
features/step_probe_probs.csv (train = out-of-fold, val/test = final probe).
Try layers -8/-4/-1 and pooling mean/last_token; pick on verified-val AUROC.

## 3. Build Consistency
    python -m signals.consistency --probs features/step_probe_probs.csv \
        --output features/consistency_scores.csv --mode ema --alpha 0.5
Definition is a DRAFT (ema | mean | max); confirm before Day 4.

## Later (Day 4+)
    python -m training.generate_features --labels training/labels_split.csv \
        --signal features/ars_scores.csv --signal features/deviation_scores.csv \
        --signal features/consistency_scores.csv --signal features/causal_scores.csv \
        --output training/split_matrix.csv
    python -m training.train_rfs --input training/split_matrix.csv
    python -m calibration.calibrate --input training/split_matrix.csv

Generated CSVs (features/*.csv, training/labels_split.csv, answer_level.csv, split_matrix.csv)
are git-ignored; share them via Drive if teammates need them.
