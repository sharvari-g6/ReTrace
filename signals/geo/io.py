"""Load per-trajectory hidden states and align them with the step-label table."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def load_trajectories(hidden_dir, labels_csv, layer=-1, pooling="mean"):
    import torch
    lab = pd.read_csv(labels_csv).sort_values(["sample_id", "step_id"])
    groups = {sid: g for sid, g in lab.groupby("sample_id")}
    trajs, seen = [], set()
    for path in sorted(Path(hidden_dir).rglob("*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        for rec in obj["records"]:
            sid = rec["sample_id"]
            if sid not in groups:
                continue
            if sid in seen:
                raise ValueError(f"{sid} appears in more than one shard.")
            seen.add(sid)
            g = groups[sid]
            n = len(g)
            if not (g["step_id"].to_numpy() == np.arange(n)).all():
                raise ValueError(f"{sid}: label steps are not a 0..n-1 prefix.")
            H = rec["step_hidden_states"][layer][pooling].float().numpy()[:n]
            if len(H) != n:
                raise ValueError(f"{sid}: {len(H)} hidden rows for {n} labelled steps.")
            trajs.append({"sample_id": sid, "problem_id": g["problem_id"].iloc[0], "split": g["split"].iloc[0],
                          "source": g["source"].iloc[0] if "source" in g else "?", "H": H,
                          "label": g["label"].to_numpy(), "verified": g["verified"].to_numpy()})
    missing = set(groups) - seen
    if missing:
        print(f"WARNING: {len(missing)} labelled trajectories have no hidden states (e.g. {sorted(missing)[:3]}).")
    return trajs
