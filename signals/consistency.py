"""Consistency signal: how much suspicion has ACCUMULATED up to step t (no future steps used).

Built from the step probe's probabilities (features/step_probe_probs.csv; train rows are
out-of-fold). DRAFT definition, confirm against the streaming-hallucination paper:
    ema : C_t = alpha * p_t + (1 - alpha) * C_{t-1},  C_0 = p_0
    mean: running mean of p_0..p_t
    max : running max of p_0..p_t
    python -m signals.consistency --probs features/step_probe_probs.csv \
        --output features/consistency_scores.csv --mode ema --alpha 0.5
"""
import argparse
import numpy as np
import pandas as pd
from features.schema import KEY_COLUMNS


def _running(v: np.ndarray, mode: str, alpha: float) -> np.ndarray:
    if mode == "mean":
        return np.cumsum(v) / np.arange(1, len(v) + 1)
    if mode == "max":
        return np.maximum.accumulate(v)
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return out


def consistency(probs: pd.DataFrame, mode="ema", alpha=0.5) -> pd.DataFrame:
    probs = probs.sort_values(KEY_COLUMNS).copy()
    probs["Consistency"] = probs.groupby("sample_id")["probe_prob"].transform(
        lambda s: pd.Series(_running(s.to_numpy(), mode, alpha), index=s.index))
    return probs[KEY_COLUMNS + ["Consistency"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probs", default="features/step_probe_probs.csv")
    p.add_argument("--output", default="features/consistency_scores.csv")
    p.add_argument("--mode", default="ema", choices=["ema", "mean", "max"])
    p.add_argument("--alpha", type=float, default=0.5)
    a = p.parse_args()
    out = consistency(pd.read_csv(a.probs), a.mode, a.alpha)
    out.to_csv(a.output, index=False)
    print(f"Wrote {a.output}: {len(out)} rows (mode={a.mode})")


if __name__ == "__main__":
    main()
