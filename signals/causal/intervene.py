"""Causal-contribution signal: does step t actually drive the model's final answer?

Adapted from the activation-patching protocol of Jin et al., "Final Checkpoints Are Not Enough"
(arXiv 2607.06648): norm-noise replacement of the hidden states (isotropic Gaussian rescaled to the
original norm, several noise trials), effect = drop in the log-probability of the model's own answer.
Here the patched states are those of the TOKENS OF STEP t, at the output of decoder layer L, while the
rest of the trajectory and the answer cue are left intact.

  prompt   = "Question: ...\\nReasoning:\\nStep 1: ...\\n...Step n: ...\\n" + cue      (full trajectory)
  answer   = greedy answer from the clean prompt (the model's own answer a*)
  effect_t = mean over R noise trials of [ log p_clean(a*) - log p_patched_t,r(a*) ]
Only raw log-probabilities are stored; the score is built in signals/causal/aggregate.py.
A positive effect means that destroying step t's information weakens the answer (step t matters).

  python -m signals.causal.intervene --data data/processed/processbench.jsonl --limit 5 \
      --output features/causal_trials_check.jsonl
  python -m signals.causal.intervene --data <jsonl> --limit 2 --tiny-random --output /tmp/c.jsonl   # CPU plumbing test
Resumable. Needs the same GPU as ARS (Colab T4 is fine).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from signals.ars.prompts import build_prompt, ANSWER_CUE
from signals.ars.answers import extract_answer
from signals.ars.generate_variants import (read_jsonl, load_done, step_seed, load_model, generate)
from signals.causal.spans import step_char_spans, token_indices


class StepPatch:
    """Forward hook on one decoder layer: replace the hidden states at `idx` (token positions)."""

    def __init__(self, idx, mode, seed):
        self.idx, self.mode, self.seed, self.fired = idx, mode, seed, False

    def __call__(self, module, inputs, output):
        import torch
        hs = output[0] if isinstance(output, tuple) else output
        sel = hs[:, self.idx, :]
        if self.mode == "zero":
            new = torch.zeros_like(sel, dtype=torch.float32)
        else:                                       # norm-noise: Gaussian direction, original norm
            g = torch.Generator(device=hs.device)
            g.manual_seed(self.seed)
            z = torch.randn(sel.shape, generator=g, device=hs.device, dtype=torch.float32)
            new = z / (z.norm(dim=-1, keepdim=True) + 1e-8) * sel.float().norm(dim=-1, keepdim=True)
        hs[:, self.idx, :] = new.to(hs.dtype)
        self.fired = True
        return output


def answer_logprob(model, ids, n_ans):
    """Sum of log-probs of the last n_ans tokens of each row (teacher forcing, no full-vocab logits)."""
    import torch
    with torch.inference_mode():
        h = model.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        logits = model.lm_head(h[:, -(n_ans + 1):-1, :]).float()
        tgt = ids[:, -n_ans:]
        lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp.sum(-1).cpu().tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--sample-list", default=None)
    p.add_argument("--output", default="features/causal_trials.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--no-quant", action="store_true")
    p.add_argument("--min-vram-gb", type=float, default=None)
    p.add_argument("--tiny-random", action="store_true")
    p.add_argument("--layer", type=int, default=24, help="decoder layer whose output is patched (Qwen3-8B has 36)")
    p.add_argument("--replace", choices=["noise", "zero"], default="noise")
    p.add_argument("--trials", type=int, default=3, help="noise trials per step (the paper uses 5)")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    import torch
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {k: getattr(a, k) for k in ("model", "no_quant", "layer", "replace", "trials", "max_new_tokens", "seed")}
    cfg_path = out_path.with_suffix(".config.json")
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        raise SystemExit(f"{cfg_path} holds different settings; use a new --output or delete both files.")
    cfg_path.write_text(json.dumps(cfg, indent=2))

    wanted = set(pd.read_csv(a.sample_list)["sample_id"].astype(str)) if a.sample_list else None
    done = load_done(out_path)
    tok, model = load_model(a)
    device = model.device
    n_layers = len(model.model.layers)
    if not 0 <= a.layer < n_layers:
        raise SystemExit(f"--layer must be in [0, {n_layers - 1}] for this model")
    layer_mod = model.model.layers[a.layer]

    t0, n_steps = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for s in read_jsonl(a.data, wanted, a.limit):
            sid, steps, q = s["sample_id"], s["steps"], s["question"]
            todo = [t for t in range(len(steps)) if (sid, t) not in done]
            if not todo:
                continue
            prompt = build_prompt(q, steps, len(steps) - 1)
            enc = tok(prompt, return_offsets_mapping=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]

            def write(rec):
                fout.write(json.dumps(rec) + "\n"); fout.flush()

            if ids.shape[1] > a.max_prompt_tokens:
                for t in todo:
                    write({"sample_id": sid, "step_id": t, "status": "too_long"})
                continue
            try:
                torch.manual_seed(step_seed(a.seed, sid, -1))
                ans_raw = generate(model, tok, ids, sample=False, temperature=None, top_p=None,
                                   max_new=a.max_new_tokens)[0]
                ans = extract_answer(ans_raw)
                if not ans:
                    for t in todo:
                        write({"sample_id": sid, "step_id": t, "status": "no_answer", "orig_raw": ans_raw})
                    continue
                ans_ids = tok(" " + ans, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                full = torch.cat([ids, ans_ids], dim=1)
                n_ans = ans_ids.shape[1]
                clean = answer_logprob(model, full, n_ans)[0]
                spans = step_char_spans(q, steps)
                for t in todo:
                    idx = token_indices(offsets, *spans[t])
                    if not idx:
                        write({"sample_id": sid, "step_id": t, "status": "no_span"}); continue
                    hook = StepPatch(torch.tensor(idx, device=device), a.replace, step_seed(a.seed, sid, t))
                    handle = layer_mod.register_forward_hook(hook)
                    try:
                        lp = answer_logprob(model, full.repeat(a.trials, 1), n_ans)
                    finally:
                        handle.remove()
                    if not hook.fired:
                        raise RuntimeError("patch hook never fired")
                    write({"sample_id": sid, "step_id": t, "status": "ok", "orig_raw": ans_raw, "answer": ans,
                           "n_ans_tokens": n_ans, "n_step_tokens": len(idx), "logp_clean": clean, "logp_patch": lp})
                    n_steps += 1
                    if n_steps % 20 == 0:
                        print(f"[{n_steps} steps | {(time.time() - t0) / n_steps:.1f}s/step]", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for t in todo:
                    write({"sample_id": sid, "step_id": t, "status": "oom"})
    print(f"Done: {n_steps} steps in {(time.time() - t0) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
