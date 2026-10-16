"""ARS stability measurement: latent-perturbation trials at every reasoning-step boundary.

For each trajectory and each step t:
  prompt   = "Question: ...\\nReasoning:\\nStep 1: ...\\n...Step t+1: ...\\n" + answer cue
  original = greedy answer from the clean prompt
  trials   = K answers where Gaussian noise is added to the LAST decoder layer's output at the
             LAST prompt token (first forward pass only), as in the official ARS code
             (radiolab-ntu/ars_icml2026: noise_scale 0.05, sampling T=0.7, top_p=0.95)
  control  = (optional, --control) K clean sampled answers: shows how much disagreement is just
             sampling noise
Only raw generated text is stored; agreement is computed later in signals/ars/aggregate.py.

Example (GPU, e.g. Colab):
  python -m signals.ars.generate_variants --data data/processed/processbench.jsonl \
      --sample-list sampled_ids.csv --output features/ars_trials.jsonl --num-trials 4
CPU smoke test of the hook/generation plumbing with random weights (needs the Qwen3 tokenizer):
  python -m signals.ars.generate_variants --data <jsonl> --limit 2 --tiny-random --output /tmp/t.jsonl
The run is resumable: finished (sample_id, step_id) pairs in --output are skipped.
"""
from __future__ import annotations
import argparse
import json
import time
import zlib
from pathlib import Path

import pandas as pd

from signals.ars.prompts import build_prompt
from signals.ars.answers import extract_answer, answers_agree


def read_jsonl(paths, wanted=None, limit=None):
    n = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                if wanted is not None and s["sample_id"] not in wanted:
                    continue
                if not s.get("steps"):
                    continue
                yield s
                n += 1
                if limit and n >= limit:
                    return


def load_done(path):
    done = set()
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["sample_id"], r["step_id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def step_seed(base_seed: int, sample_id: str, step_id: int) -> int:
    return (base_seed * 1_000_003 + zlib.crc32(f"{sample_id}:{step_id}".encode())) % (2**31 - 1)


def check_gpu(args):
    """Fail fast (before the 16 GB download) when the real model cannot fit on a GPU."""
    import torch
    if args.tiny_random:
        return
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU is visible to PyTorch (torch %s, CUDA build: %s).\n"
            "Qwen3-8B in 4-bit needs an NVIDIA GPU with ~8 GB free. Options: run this on Colab with a GPU "
            "runtime (as the extraction notebook did), or install a CUDA build of torch. "
            "Use --tiny-random to test the code on CPU." % (torch.__version__, torch.version.cuda))
    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / 1e9
    print(f"GPU: {props.name}, {gb:.1f} GB")
    need = args.min_vram_gb if args.min_vram_gb is not None else (8.0 if not args.no_quant else 18.0)
    if gb < need:
        raise SystemExit(f"{props.name} has {gb:.1f} GB; Qwen3-8B needs about {need:.0f} GB "
                         f"({'4-bit' if not args.no_quant else 'fp16'}). Use a bigger GPU (Colab) instead.")


def load_model(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    check_gpu(args)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.tiny_random:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        cfg.hidden_size, cfg.intermediate_size = 64, 128
        cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads = 2, 4, 2
        cfg.head_dim = 16
        model = AutoModelForCausalLM.from_config(cfg)
        return tok, model.eval()
    kw = dict(device_map="auto", torch_dtype=torch.float16)   # works on Colab's transformers; newer ones only warn
    if not args.no_quant:                      # same 4-bit NF4 setup as the extraction notebook
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    return tok, model.eval()


def last_layer(model):
    return model.model.layers[-1]


class LastTokenNoise:
    """Forward hook: add Gaussian noise to the last decoder layer's output at the last prompt
    position, on the first forward pass (prefill) only, independently for each batch row."""

    def __init__(self, scale, mode, seed):
        self.scale, self.mode, self.seed, self.done = scale, mode, seed, False

    def __call__(self, module, inputs, output):
        import torch
        if self.done:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        last = hs[:, -1, :]
        g = torch.Generator(device=hs.device)
        g.manual_seed(self.seed)
        z = torch.randn(last.shape, generator=g, device=hs.device, dtype=torch.float32)
        if self.mode == "relative":            # noise proportional to the state's RMS
            scale = self.scale * last.float().pow(2).mean(-1, keepdim=True).sqrt()
        else:                                   # absolute, as in the paper's code
            scale = self.scale
        hs[:, -1, :] = (last.float() + scale * z).to(hs.dtype)
        self.done = True
        return output


def generate(model, tok, ids, *, sample, temperature, top_p, max_new, hook=None):
    import torch
    handle = last_layer(model).register_forward_hook(hook) if hook is not None else None
    try:
        kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        if sample:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kw)
    finally:
        if handle is not None:
            handle.remove()
    texts = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
    return [t.split("\n")[0].strip() for t in texts]       # answer = first line only


def embed_last_token(model, tok, prompt, answers, layers, device):
    """Last-token hidden states of (prompt + answer) for each answer; layers use HF
    hidden_states indexing (-1 = last), same convention as the extraction notebook."""
    import torch
    texts = [prompt + " " + a for a in answers]
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    lens = enc["attention_mask"].sum(1)
    with torch.inference_mode():
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
    rows = torch.arange(len(texts), device=device)
    # right-pad so the last real token index is len-1 (tokenizer padding side may differ)
    if tok.padding_side == "left":
        idx = torch.full_like(lens, enc["input_ids"].shape[1] - 1)
    else:
        idx = lens - 1
    return {L: hs[L][rows, idx].float().cpu().numpy().astype("float16") for L in layers}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True, help="processed jsonl file(s) with question/steps")
    p.add_argument("--sample-list", default=None, help="CSV with sample_id column (the pilot set)")
    p.add_argument("--output", default="features/ars_trials.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--no-quant", action="store_true")
    p.add_argument("--min-vram-gb", type=float, default=None,
                   help="override the GPU-memory check (e.g. 3 for a small dev model such as Qwen/Qwen3-1.7B)")
    p.add_argument("--tiny-random", action="store_true", help="random tiny model for a CPU plumbing test")
    p.add_argument("--num-trials", type=int, default=4)
    p.add_argument("--noise-scale", type=float, default=0.05)
    p.add_argument("--noise-mode", choices=["absolute", "relative"], default="absolute")
    p.add_argument("--greedy-trials", action="store_true",
                   help="decode trials greedily so any change is caused by the noise alone")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument("--control", action="store_true", help="also K clean sampled answers per step")
    p.add_argument("--save-embeddings", default=None, help="dir for per-trajectory .npz (for shaping)")
    p.add_argument("--emb-layers", type=int, nargs="+", default=[-8, -4, -1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="max trajectories (smoke tests)")
    a = p.parse_args()

    import torch
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in vars(a).items() if k in
           ("model", "no_quant", "num_trials", "noise_scale", "noise_mode", "greedy_trials", "temperature",
            "top_p", "max_new_tokens", "control", "seed")}
    cfg_path = out_path.with_suffix(".config.json")
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        raise SystemExit(f"{cfg_path} holds different settings; use a new --output or delete both files.")
    cfg_path.write_text(json.dumps(cfg, indent=2))

    wanted = set(pd.read_csv(a.sample_list)["sample_id"].astype(str)) if a.sample_list else None
    done = load_done(out_path)
    tok, model = load_model(a)
    device = model.device
    sample_trials = not a.greedy_trials
    if a.save_embeddings:
        Path(a.save_embeddings).mkdir(parents=True, exist_ok=True)
        import numpy as np

    t0, n_steps, n_agree_seen, n_valid_seen = time.time(), 0, 0, 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for s in read_jsonl(a.data, wanted, a.limit):
            sid, steps = s["sample_id"], s["steps"]
            emb_rows = {L: [] for L in a.emb_layers}
            emb_meta = []
            for t in range(len(steps)):
                if (sid, t) in done:
                    continue
                prompt = build_prompt(s["question"], steps, t)
                ids = tok(prompt, return_tensors="pt").input_ids.to(device)
                rec = {"sample_id": sid, "step_id": t, "status": "ok"}
                if ids.shape[1] > a.max_prompt_tokens:
                    rec["status"] = "too_long"
                    fout.write(json.dumps(rec) + "\n")
                    continue
                try:
                    seed = step_seed(a.seed, sid, t)
                    torch.manual_seed(seed)
                    orig = generate(model, tok, ids, sample=False, temperature=None, top_p=None,
                                    max_new=a.max_new_tokens)[0]
                    hook = LastTokenNoise(a.noise_scale, a.noise_mode, seed)
                    batch = ids.repeat(a.num_trials, 1)
                    trials = generate(model, tok, batch, sample=sample_trials, temperature=a.temperature,
                                      top_p=a.top_p, max_new=a.max_new_tokens, hook=hook)
                    if not hook.done:
                        raise RuntimeError("noise hook never fired (check the decoder-layer path)")
                    rec["orig_raw"] = orig
                    rec["trials"] = [{"k": k, "raw": x} for k, x in enumerate(trials)]
                    if a.control:
                        torch.manual_seed(seed + 1)
                        ctrl = generate(model, tok, batch, sample=True, temperature=a.temperature,
                                        top_p=a.top_p, max_new=a.max_new_tokens)
                        rec["control"] = [{"k": k, "raw": x} for k, x in enumerate(ctrl)]
                    if a.save_embeddings:
                        answers = [orig] + trials
                        emb = embed_last_token(model, tok, prompt, answers, a.emb_layers, device)
                        for L in a.emb_layers:
                            emb_rows[L].append(emb[L])
                        emb_meta += [(t, k) for k in range(-1, a.num_trials)]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rec = {"sample_id": sid, "step_id": t, "status": "oom"}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_steps += 1
                if rec["status"] == "ok":
                    o = extract_answer(rec["orig_raw"])
                    for tr in rec["trials"]:
                        v = answers_agree(o, extract_answer(tr["raw"])) if o else None
                        if v is not None:
                            n_valid_seen += 1
                            n_agree_seen += int(v)
                if n_steps % 20 == 0:
                    rate = n_agree_seen / max(n_valid_seen, 1)
                    print(f"[{n_steps} steps | {(time.time() - t0) / n_steps:.1f}s/step | "
                          f"running agreement {rate:.2f}]", flush=True)
            if a.save_embeddings and emb_meta:
                meta = np.array(emb_meta)
                np.savez(Path(a.save_embeddings) / f"{sid}.npz", step_ids=meta[:, 0], ks=meta[:, 1],
                         **{f"emb_{'m' + str(-L)}": np.concatenate(emb_rows[L]) for L in a.emb_layers})
    print(f"Done: {n_steps} new steps in {(time.time() - t0) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
