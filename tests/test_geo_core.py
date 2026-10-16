"""CPU tests for the Deviation (GeoReason-inspired) module. Run: python -m tests.test_geo_core"""
import os, tempfile
import numpy as np
from sklearn.metrics import roc_auc_score
from signals.geo import geometry as G


def synth(n_traj=120, d=64, seed=0, offset=0.3):
    rng = np.random.default_rng(seed)
    shift = np.random.default_rng(1234).normal(size=d); shift /= np.linalg.norm(shift)   # same error direction for every seed
    trajs = []
    for i in range(n_traj):
        n = int(rng.integers(4, 9))
        fe = int(rng.integers(1, n)) if rng.random() < 0.6 else n
        base = rng.normal(size=d) * offset                    # per-trace nuisance offset
        H = base + rng.normal(size=(n, d)) * 0.5
        lab = (np.arange(n) >= fe).astype(int)
        ver = (np.arange(n) <= fe).astype(int) if fe < n else np.ones(n, int)
        H[lab == 1] += 3.0 * shift
        trajs.append({"H": H, "label": lab, "verified": ver})
    return trajs


def test_phi_zero_padding():
    Z = np.arange(1, 7, dtype=float).reshape(3, 2)
    P = G.phi_features(Z)
    assert P.shape == (3, 6)
    assert np.allclose(P[0], [1, 2, 1, 2, 1, 2])              # dz0 = z0, d2z0 = z0
    assert np.allclose(P[1, 2:4], Z[1] - Z[0])
    assert np.allclose(P[2, 4:], Z[2] - 2 * Z[1] + Z[0])


def test_signal_found_and_label_free():
    tr, ev = synth(seed=0), synth(seed=1)
    m = G.fit(tr, G.Config(k=4))
    s = np.concatenate([G.score(m, t["H"]) for t in ev])
    y = np.concatenate([t["label"] for t in ev])
    assert roc_auc_score(y, s) > 0.9
    t = ev[0]                                                   # scoring must not depend on labels
    assert np.allclose(G.score(m, t["H"], t["label"]), G.score(m, t["H"], 1 - t["label"]))
    assert np.isfinite(s).all() and (s >= 0).all()


def test_oracle_removes_trace_offset_but_global_cannot():
    # large per-trace offsets: the label-conditioned (oracle) normalisation recovers the signal,
    # the deployable global normalisation does not. This is why 'oracle' is diagnostic only.
    tr, ev = synth(seed=0, offset=2.0), synth(seed=1, offset=2.0)
    y = np.concatenate([t["label"] for t in ev])
    aucs = {}
    for center in ("global", "oracle"):
        m = G.fit(tr, G.Config(k=4, center=center))
        s = np.concatenate([G.score(m, t["H"], t["label"]) for t in ev])
        aucs[center] = roc_auc_score(y, s)
    assert aucs["oracle"] > 0.8 and aucs["oracle"] > aucs["global"] + 0.15, aucs


def test_save_load_roundtrip():
    tr, ev = synth(seed=2), synth(n_traj=5, seed=3)
    m = G.fit(tr, G.Config(k=4, cost="diag"))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "R0.npz"); G.save(m, p); m2 = G.load(p)
    for t in ev:
        assert np.allclose(G.score(m, t["H"]), G.score(m2, t["H"]))


def test_too_few_errors_raises():
    tr = synth(n_traj=3, seed=4)
    for t in tr:
        t["label"][:] = 0; t["verified"][:] = 1
    try:
        G.fit(tr, G.Config(k=4)); raise AssertionError("should have raised")
    except (ValueError, RuntimeError):
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
