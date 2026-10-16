# ARS (stability signal): run guide

Run from the repo root on a GPU machine (Colab T4/A100 is fine). `pip install bitsandbytes accelerate`
if missing. The input is the processed jsonl (question, steps, sample_id) used for extraction.

## Where to run
Needs an NVIDIA GPU with ~8 GB free (4-bit). On a laptop without one, use Colab (GPU runtime):
    from google.colab import drive; drive.mount('/content/drive')
    !git clone -b sharvari https://github.com/sharvari-g6/ReTrace.git && cd ReTrace   # then unzip the ARS files over it
    !pip install -q -U transformers accelerate bitsandbytes
    # data: unzip /content/drive/MyDrive/ReTrace/retrace_colab_data.zip (as in the extraction notebook)
    !python -m signals.ars.generate_variants --data /content/retrace_data/processed/processbench.jsonl \
        --limit 5 --control --output /content/drive/MyDrive/ReTrace/ars/ars_trials_check.jsonl
Write --output to Drive so the run survives a disconnect (it resumes). The script now exits within seconds
with a clear message if no suitable GPU is found, before downloading anything.

## Optional local dev run on a small GPU (6 GB), a DIFFERENT, smaller model, only to check the plumbing
Needs a CUDA build of torch (pip install torch --index-url https://download.pytorch.org/whl/cu124, pick your CUDA version) and bitsandbytes.
    python -m signals.ars.generate_variants --data data/processed/processbench.jsonl --limit 3 --control \
        --model Qwen/Qwen3-1.7B --min-vram-gb 3 --output features/dev_trials.jsonl
Do not use these outputs for the project: the real signal must come from Qwen3-8B, the model whose hidden states were extracted.

## 0. CPU checks (no GPU, no model)
    python -m tests.test_ars_core                       # 5 tests print "ok"
    python -m signals.ars.generate_variants --data data/processed/processbench.jsonl \
        --limit 2 --tiny-random --num-trials 2 --output /tmp/smoke.jsonl     # plumbing only, random weights

## 1. Real run, small first (5 trajectories), read the output before scaling up
    python -m signals.ars.generate_variants --data data/processed/processbench.jsonl --limit 5 \
        --num-trials 4 --control --output features/ars_trials_check.jsonl
    python -m signals.ars.aggregate --trials features/ars_trials_check.jsonl \
        --output features/ars_check.csv --diagnostics features/ars_check_diag.csv
Open the jsonl: orig_raw should be a short answer, trials should be mostly sensible and sometimes
different. If every step has ARS = 0, the noise is too weak (try --noise-mode relative --noise-scale 0.05
or --greedy-trials off/on, see below). If everything disagrees, the answer cue/extraction is off.

## 2. Pilot run (the agreed ~800-trajectory ARS/Causal subset)
    python -m signals.ars.generate_variants --data data/processed/processbench.jsonl data/processed/prm800k.jsonl \
        --sample-list sampled_ids_ars.csv --num-trials 4 --control --output features/ars_trials.jsonl
Resumable: rerun the same command after a disconnect. Settings are locked in features/ars_trials.config.json.

## 3. Scores and report
    python -m signals.ars.aggregate --trials features/ars_trials.jsonl
    python -m signals.ars.report --labels training/labels_split.csv
Outputs: features/ars_scores.csv (sample_id, step_id, ARS), features/ars_diagnostics.csv, artifacts/ars_report.json.

## Optional: representation shaping (the paper's learning step)
    ... generate_variants ... --save-embeddings features/ars_emb
    python -m signals.ars.shaping train   --emb-dir features/ars_emb --trials features/ars_trials.jsonl --labels training/labels_split.csv
    python -m signals.ars.shaping project --emb-dir features/ars_emb
