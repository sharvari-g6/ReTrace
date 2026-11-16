# Deviation (GeoReason-inspired) and Causal signals: run guide

## Deviation: CPU only, uses the extracted hidden-state shards
    python -m tests.test_geo_core                                   # 5 "ok" lines
    python -m training.build_R0 --hidden-dir train_features_raw --labels training/labels_split.csv
Writes features/deviation_scores.csv (sample_id, step_id, Deviation), models/R0.npz (final reference geometry),
artifacts/deviation_report.json. Train rows are scored out-of-fold (shared folds), val/test with the model fit on all
train. Defaults follow the paper (k=16, alpha=1, rho=0.25, last layer, mean pooling); do not tune them on a tiny sample.

## Causal: GPU (Colab), same setup as ARS (see ARS_RUN.md for Drive/clone/pip steps)
    python -m tests.test_causal_core
    python -m signals.causal.intervene --data /content/retrace_data/processed/processbench.jsonl --limit 5 \
        --output /content/drive/MyDrive/ReTrace_ars/causal_trials_check.jsonl
    python -m signals.causal.aggregate --trials <that file> --output <csv> --diagnostics <csv>
Pilot: add --sample-list sampled_ids_ars.csv (the same ~800 trajectories as ARS) and a new --output. Resumable.
Cost: 1 greedy answer + 1 clean pass per trajectory, then `--trials` (default 3) patched passes per step.
CPU plumbing test (random weights): add --tiny-random --limit 2 (needs only the tokenizer download).
