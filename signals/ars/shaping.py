"""OPTIONAL: ARS representation shaping (the paper's learning step) on saved embeddings.

Needs the per-trajectory .npz files written by generate_variants.py --save-embeddings and the
trial log. For every (trajectory, step) the original and its K noisy variants form one group;
each variant is labelled "agree" / "disagree" with the original by the equivalence rule
(the original counts as "agree"). A linear map d -> 128 is trained with the paper's objective:
positives = same group, same label; negatives = same group other label + all other groups.
Implementation notes (deviations from the official code): batches are made of whole groups
(the official loader mixes random samples), the loss is vectorised, and training uses TRAIN-split
trajectories only. No ground-truth step labels are used here.

  python -m signals.ars.shaping train   --emb-dir features/ars_emb --trials features/ars_trials.jsonl \
        --labels training/labels_split.csv --layer -1
  python -m signals.ars.shaping project --emb-dir features/ars_emb --layer -1
`project` writes features/ars_projected_L{layer}.npz (sample_id, step_id, proj) for the ORIGINAL
answers of every step; turning that into a score needs a probe (see the decision note).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from signals.ars.aggregate import load_trials
from signals.ars.answers import extract_answer, answers_agree


def _key(layer: int) -> str:
    return f"emb_m{-layer}"


def load_groups(emb_dir, trials_path, layer, keep_sample_ids=None):
    """Returns X (n,d), group (n,), consistent (n,) bool, plus row meta."""
    recs = {(r["sample_id"], r["step_id"]): r for r in load_trials(trials_path)}
    X, group, cons, meta = [], [], [], []
    gid = 0
    for path in sorted(Path(emb_dir).glob("*.npz")):
        sid = path.stem
        if keep_sample_ids is not None and sid not in keep_sample_ids:
            continue
        z = np.load(path)
        emb, steps, ks = z[_key(layer)], z["step_ids"], z["ks"]
        for t in sorted(set(steps.tolist())):
            r = recs.get((sid, t))
            if r is None or r.get("status") != "ok":
                continue
            orig = extract_answer(r["orig_raw"])
            if orig is None:
                continue
            idx = np.where(steps == t)[0]
            rows = [(i, True) for i in idx if ks[i] == -1]
            for i in idx:
                if ks[i] >= 0:
                    v = answers_agree(orig, extract_answer(r["trials"][int(ks[i])]["raw"]))
                    if v is not None:
                        rows.append((i, bool(v)))
            if len(rows) < 2:
                continue
            for i, c in rows:
                X.append(emb[i]); group.append(gid); cons.append(c); meta.append((sid, int(t), int(ks[i])))
            gid += 1
    return np.asarray(X, dtype="float32"), np.asarray(group), np.asarray(cons), meta


def ars_loss(h, group, cons, tau=0.1):
    import torch
    h = torch.nn.functional.normalize(h, dim=1)
    sim = h @ h.T / tau
    n = sim.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    same_g = group[:, None] == group[None, :]
    same_l = cons[:, None] == cons[None, :]
    pos = same_g & same_l & ~eye
    neg_inf = torch.finfo(sim.dtype).min
    den = torch.logsumexp(sim.masked_fill(eye, neg_inf), dim=1)
    num = torch.logsumexp(sim.masked_fill(~pos, neg_inf), dim=1)
    ok = pos.any(1)
    return (den - num)[ok].mean() if ok.any() else sim.sum() * 0.0


def train(a):
    import torch
    lab = pd.read_csv(a.labels)[["sample_id", "split"]].drop_duplicates()
    train_ids = set(lab.loc[lab["split"] == "train", "sample_id"])
    X, group, cons, _ = load_groups(a.emb_dir, a.trials, a.layer, train_ids)
    if len(X) == 0:
        raise SystemExit("No training groups found (check --emb-dir, --trials and --labels).")
    print(f"{len(np.unique(group))} groups, {len(X)} embeddings, agree rate {cons.mean():.2f}")
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.nn.Linear(X.shape[1], a.out_dim, bias=False).to(dev)
    opt = torch.optim.Adam(W.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    Xt, Gt, Ct = (torch.from_numpy(X).to(dev), torch.from_numpy(group).to(dev), torch.from_numpy(cons).to(dev))
    uniq = np.unique(group)
    rng = np.random.default_rng(a.seed)
    for ep in range(a.epochs):
        rng.shuffle(uniq)
        tot, nb = 0.0, 0
        for i in range(0, len(uniq), a.groups_per_batch):
            sel = torch.isin(Gt, torch.from_numpy(uniq[i:i + a.groups_per_batch]).to(dev))
            loss = ars_loss(W(Xt[sel]), Gt[sel], Ct[sel], a.tau) + a.lambda_reg * (W.weight ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep + 1}/{a.epochs} loss {tot / max(nb, 1):.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"W": W.state_dict(), "layer": a.layer, "in_dim": X.shape[1], "out_dim": a.out_dim}, a.out)
    print("saved", a.out)


def project(a):
    import torch
    ck = torch.load(a.model, map_location="cpu")
    W = torch.nn.Linear(ck["in_dim"], ck["out_dim"], bias=False)
    W.load_state_dict(ck["W"])
    sids, steps, proj = [], [], []
    for path in sorted(Path(a.emb_dir).glob("*.npz")):
        z = np.load(path)
        m = z["ks"] == -1
        with torch.no_grad():
            p = W(torch.from_numpy(z[_key(ck["layer"])][m]).float()).numpy()
        sids += [path.stem] * int(m.sum()); steps += z["step_ids"][m].tolist(); proj.append(p)
    out = Path(a.out_proj or f"features/ars_projected_L{-ck['layer']}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, sample_id=np.array(sids), step_id=np.array(steps), proj=np.concatenate(proj))
    print("saved", out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--emb-dir", required=True); t.add_argument("--trials", required=True)
    t.add_argument("--labels", required=True); t.add_argument("--layer", type=int, default=-1)
    t.add_argument("--out", default="artifacts/ars_linear_map.pt")
    t.add_argument("--out-dim", type=int, default=128); t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--tau", type=float, default=0.1); t.add_argument("--lambda-reg", type=float, default=1e-3)
    t.add_argument("--groups-per-batch", type=int, default=8); t.add_argument("--seed", type=int, default=42)
    t.set_defaults(fn=train)
    q = sub.add_parser("project")
    q.add_argument("--emb-dir", required=True); q.add_argument("--model", default="artifacts/ars_linear_map.pt")
    q.add_argument("--layer", type=int, default=-1); q.add_argument("--out-proj", default=None)
    q.set_defaults(fn=project)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
