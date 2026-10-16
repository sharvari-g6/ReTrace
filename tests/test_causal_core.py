"""CPU tests for the Causal signal's model-free logic. Run: python -m tests.test_causal_core"""
import json, os, tempfile
import numpy as np
from signals.causal.spans import step_char_spans, token_indices
from signals.causal.aggregate import build, load
from signals.ars.prompts import build_prefix


def test_spans_tile_the_prefix():
    q, steps = "What is 2+2?", ["Add them.", "It is 4.", "Done."]
    spans = step_char_spans(q, steps)
    full = build_prefix(q, steps, len(steps) - 1)
    assert spans[0][0] == len(build_prefix(q, steps, -1)) and spans[-1][1] == len(full)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))     # contiguous, no gaps
    assert full[spans[1][0]:spans[1][1]] == "Step 2: It is 4.\n"


def test_token_indices():
    offsets = [(0, 0), (0, 5), (5, 9), (9, 14), (14, 20)]           # first token is special (empty span)
    assert token_indices(offsets, 5, 14) == [2, 3]
    assert token_indices(offsets, 0, 5) == [1]
    assert token_indices(offsets, 30, 40) == []


def test_aggregate_effect_math():
    recs = [{"sample_id": "A", "step_id": 0, "status": "ok", "logp_clean": -1.0, "logp_patch": [-4.0, -2.0, -3.0], "n_ans_tokens": 2},
            {"sample_id": "A", "step_id": 1, "status": "ok", "logp_clean": -1.0, "logp_patch": [-0.5, -0.5], "n_ans_tokens": 1},
            {"sample_id": "A", "step_id": 2, "status": "ok", "logp_clean": -1.0, "logp_patch": [-500.0], "n_ans_tokens": 1},
            {"sample_id": "A", "step_id": 3, "status": "no_answer"},
            {"sample_id": "A", "step_id": 4, "status": "ok", "logp_clean": -1.0, "logp_patch": [float("nan")], "n_ans_tokens": 1}]
    scores, diag = build(recs, clip=20.0)
    got = dict(zip(scores.step_id, scores.Causal))
    assert np.isclose(got[0], 2.0)            # mean of (3, 1, 2)
    assert np.isclose(got[1], -0.5)           # patching made the answer MORE likely: negative effect kept
    assert np.isclose(got[2], 20.0)           # fixed cap
    assert 3 not in got and 4 not in got      # missing / non-finite steps are not invented
    assert len(diag) == 5 and list(scores.columns) == ["sample_id", "step_id", "Causal"]


def test_load_last_write_wins():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"sample_id": "A", "step_id": 0, "status": "oom"}) + "\n")
            f.write(json.dumps({"sample_id": "A", "step_id": 0, "status": "ok", "logp_clean": 0, "logp_patch": [-1]}) + "\n")
        r = load(p)
        assert len(r) == 1 and r[0]["status"] == "ok"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
