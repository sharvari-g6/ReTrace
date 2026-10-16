"""GeoReason-inspired transport-instability score (the Deviation signal). numpy/scipy only.

Follows Alvarez & Baheri, "Where Does Reasoning Break?" (arXiv 2605.13772), Sec. 3.2-3.3 and App. A:
  h_t      step hidden state (mean-pooled tokens of the step, one layer)
  h~_t     normalised state                                   (see `normalize`)
  z_t      U^T h~_t, U = top-k eigvecs of  M = (mu1-mu0)(mu1-mu0)^T + C1 - alpha*C0   (cPCA lens)
  phi_t    [z_t, dz_t, d2z_t]  with zero padding for missing previous steps (paper Eq. 6, 12)
  R0       cloud of phi over CORRECT steps of training trajectories (mean mu_R, covariance S_R)
  Deviation(t) = sqrt((phi_t - mu_R)^T A (phi_t - mu_R)),  A = I | diag(1/var) | S_R^-1
             = the point-to-cloud transport cost of App. A (Lemma A.1) without its constant term.
Labels are used ONLY to fit U, the normalisation constants and R0 on training trajectories
(never to score a trajectory), so the score is deployable. The paper's label-conditioned
"teacher" (normalising each trace by its own correct prefix) is available as center='oracle'
for DIAGNOSTICS ONLY: it needs labels at scoring time.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

EPS = 1e-6


@dataclass
class Config:
    k: int = 16              # cPCA rank (paper default)
    alpha: float = 1.0       # background penalty (paper default)
    rho: float = 0.25        # weight of post-error steps in the "error" class (paper default)
    center: str = "global"   # global | trace_mean | oracle(diagnostic)
    cost: str = "maha"       # euclid | diag | maha
    ridge: float = 1e-2      # shrinkage of the R0 covariance (relative to mean variance)


@dataclass
class Model:
    cfg: Config
    mu_g: np.ndarray | None = None   # global centre (mean of correct training states)
    sigma_g: float = 1.0             # global RMS scale
    U: np.ndarray | None = None      # (d, k)
    mu_R: np.ndarray | None = None   # (3k,)
    A: np.ndarray | None = None      # (3k, 3k) cost matrix
    info: dict = field(default_factory=dict)


def normalize(H: np.ndarray, cfg: Config, mu_g, sigma_g, labels=None) -> np.ndarray:
    """H: (n_steps, d) states of ONE trajectory."""
    H = H.astype(np.float64)
    if cfg.center == "global":
        return (H - mu_g) / (sigma_g + EPS)
    if cfg.center == "trace_mean":                      # label-free, trace-specific (post-hoc: uses all steps)
        c = H.mean(0)
        s = np.sqrt(((H - c) ** 2).mean()) + EPS
        return (H - c) / s
    if cfg.center == "oracle":                          # paper's teacher normalisation: NEEDS LABELS
        ok = np.asarray(labels) == 0
        if ok.sum() < 2:
            ok = np.ones(len(H), bool)
        c = H[ok].mean(0)
        s = np.sqrt(((H[ok] - c) ** 2).mean()) + EPS
        return (H - c) / s
    raise ValueError(cfg.center)


def phi_features(Z: np.ndarray) -> np.ndarray:
    """Z: (n_steps, k) -> (n_steps, 3k) = [z, dz, d2z] with zero padding (paper Eq. 6)."""
    z1 = np.vstack([np.zeros((1, Z.shape[1])), Z[:-1]])
    z2 = np.vstack([np.zeros((2, Z.shape[1])), Z[:-2]])[: len(Z)]
    return np.hstack([Z, Z - z1, Z - 2 * z1 + z2])


def _cpca(X0, X1, w1, alpha, k):
    """Top-k eigenvectors of M without forming the d x d matrix (matrix-free)."""
    d = X0.shape[1]
    mu0 = X0.mean(0)
    sw = w1.sum()
    mu1 = (w1[:, None] * X1).sum(0) / sw
    X0c, X1c = X0 - mu0, X1 - mu1
    X1w = X1c * np.sqrt(w1)[:, None]
    dm = mu1 - mu0

    def matvec(v):
        v = np.asarray(v).reshape(d, -1)
        out = dm[:, None] * (dm @ v) + X1w.T @ (X1w @ v) / sw - alpha * (X0c.T @ (X0c @ v)) / len(X0c)
        return out

    op = LinearOperator((d, d), matvec=matvec, matmat=matvec, dtype=np.float64)
    vals, vecs = eigsh(op, k=k, which="LA")
    order = np.argsort(-vals)
    return vecs[:, order], vals[order]


def fit(trajs: list, cfg: Config) -> Model:
    """trajs: list of dicts with H (n,d), label (n,), verified (n,)  [TRAIN trajectories only]."""
    m = Model(cfg)
    correct = np.vstack([t["H"][t["label"] == 0] for t in trajs if (t["label"] == 0).any()]).astype(np.float64)
    m.mu_g = correct.mean(0)
    m.sigma_g = float(np.sqrt(((correct - m.mu_g) ** 2).mean()))
    X0, X1, W1 = [], [], []
    for t in trajs:
        Hn = normalize(t["H"], cfg, m.mu_g, m.sigma_g, t["label"])
        lab, ver = t["label"], t["verified"]
        X0.append(Hn[lab == 0])
        first = (lab == 1) & (ver == 1)
        post = (lab == 1) & (ver == 0)
        X1.append(Hn[first]); W1.append(np.ones(first.sum()))
        if cfg.rho > 0:
            X1.append(Hn[post]); W1.append(np.full(post.sum(), cfg.rho))
    X0, X1, W1 = np.vstack(X0), np.vstack(X1), np.concatenate(W1)
    if len(X1) < 3 or len(X0) < 3:
        raise ValueError("Too few first-error / correct steps to fit the cPCA lens.")
    k = min(cfg.k, len(X1) - 1)
    m.U, vals = _cpca(X0, X1, W1, cfg.alpha, k)
    # reference cloud R0: phi over CORRECT steps, computed trajectory by trajectory
    phis = []
    for t in trajs:
        Hn = normalize(t["H"], cfg, m.mu_g, m.sigma_g, t["label"])
        P = phi_features(Hn @ m.U)
        phis.append(P[t["label"] == 0])
    R = np.vstack(phis)
    m.mu_R = R.mean(0)
    S = np.cov(R, rowvar=False) + np.eye(R.shape[1]) * cfg.ridge * np.trace(np.cov(R, rowvar=False)) / R.shape[1]
    if cfg.cost == "euclid":
        m.A = np.eye(R.shape[1])
    elif cfg.cost == "diag":
        m.A = np.diag(1.0 / np.diag(S))
    else:
        m.A = np.linalg.inv(S)
    m.info = {"top_eigenvalues": vals[:4].tolist(), "n_correct_steps": int(len(X0)),
              "n_error_steps": int(len(X1)), "k": int(k)}
    return m


def score(m: Model, H: np.ndarray, labels=None) -> np.ndarray:
    """Deviation per step of one trajectory. `labels` only used when center='oracle' (diagnostic)."""
    Hn = normalize(H, m.cfg, m.mu_g, m.sigma_g, labels)
    P = phi_features(Hn @ m.U) - m.mu_R
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", P, m.A, P), 0.0))


def save(m: Model, path: str) -> None:
    np.savez(path, U=m.U, mu_g=m.mu_g, sigma_g=m.sigma_g, mu_R=m.mu_R, A=m.A,
             cfg=np.array([m.cfg.k, m.cfg.alpha, m.cfg.rho, m.cfg.ridge]),
             center=m.cfg.center, cost=m.cfg.cost)


def load(path: str) -> Model:
    z = np.load(path, allow_pickle=False)
    k, alpha, rho, ridge = z["cfg"].tolist()
    m = Model(Config(int(k), alpha, rho, str(z["center"]), str(z["cost"]), ridge))
    m.U, m.mu_g, m.sigma_g, m.mu_R, m.A = z["U"], z["mu_g"], float(z["sigma_g"]), z["mu_R"], z["A"]
    return m
