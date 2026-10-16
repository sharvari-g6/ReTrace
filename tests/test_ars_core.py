"""CPU-only tests for the ARS logic that does not need a model. Run: python -m tests.test_ars_core"""
import json, tempfile, os
import numpy as np
import pandas as pd
from signals.ars.answers import extract_answer, answers_agree, normalize_answer
from signals.ars.prompts import build_prefix, build_prompt, ANSWER_CUE
from signals.ars.aggregate import build_scores, load_trials, step_stats
from signals.ars.generate_variants import step_seed, read_jsonl, load_done


def test_extract():
    assert extract_answer("  18 dollars.\nextra") == "18 dollars"
    assert extract_answer("The answer is 42.") == "42"
    assert extract_answer("so the final answer is \\boxed{\\frac{3}{4}} ok") == "\\frac{3}{4}"
    assert extract_answer("\\boxed{x^{2}+1}") == "x^{2}+1"
    assert extract_answer("   ") is None and extract_answer(None) is None
    assert extract_answer("\\boxed{}") is None      # an empty box is an invalid trial


def test_agree():
    assert answers_agree("18", "18.0") is True
    assert answers_agree("1,000", "1000") is True
    assert answers_agree("\\frac{3}{4}", "0.75") is True
    assert answers_agree("\\dfrac{1}{2}", "1/2") is True
    assert answers_agree("$5$", "5 dollars") is True
    assert answers_agree("90^\\circ", "90") is True
    assert answers_agree("18", "81") is False
    assert answers_agree("2x", "2") is False          # a lone letter is not a unit
    assert answers_agree("x+1", "x + 1") is True
    assert answers_agree("x+1", "x+2") is False
    assert answers_agree(None, "3") is None and answers_agree("3", None) is None
    assert normalize_answer("\\text{ 12 }") == "12"


def test_prompt():
    steps = ["a", "b", "c"]
    assert build_prefix("Q?", steps, 0) == "Question: Q?\nReasoning:\nStep 1: a\n"
    assert build_prefix("Q?", steps, 1).endswith("Step 2: b\n")
    assert "c" not in build_prefix("Q?", steps, 1)        # later steps never leak into step t's prompt
    assert build_prompt("Q?", steps, 2).endswith(ANSWER_CUE)


def _rec(sid, t, orig, trials, control=None, status="ok"):
    r = {"sample_id": sid, "step_id": t, "status": status, "orig_raw": orig,
         "trials": [{"k": i, "raw": x} for i, x in enumerate(trials)]}
    if control is not None:
        r["control"] = [{"k": i, "raw": x} for i, x in enumerate(control)]
    return r


def test_aggregate():
    recs = [_rec("A", 0, "5", ["5", "5", "5", "5"]),            # stable  -> 0
            _rec("A", 1, "5", ["5", "6", "7", "5"], ["5"] * 4),  # 2/4 agree -> 0.5
            _rec("A", 2, "5", ["6", "7", "8", "9"]),            # unstable -> 1
            _rec("A", 3, "5", ["", "", "5", ""]),               # 1 valid  -> dropped (min_valid 2)
            _rec("A", 4, "", ["5", "5"]),                       # original invalid -> dropped
            _rec("A", 5, "5", [], status="too_long")]
    scores, diag = build_scores(recs, min_valid=2)
    assert scores["ARS"].tolist() == [0.0, 0.5, 1.0] and scores["step_id"].tolist() == [0, 1, 2]
    assert len(diag) == 6 and diag.loc[diag.step_id == 1, "ARS_control"].iloc[0] == 0.0
    assert list(scores.columns) == ["sample_id", "step_id", "ARS"]      # exactly one signal column


def test_resume_and_seed():
    assert step_seed(42, "PB_00001", 3) == step_seed(42, "PB_00001", 3)
    assert step_seed(42, "PB_00001", 3) != step_seed(42, "PB_00001", 4)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps(_rec("A", 0, "5", ["5"])) + "\n" + "{broken\n")
            f.write(json.dumps(_rec("A", 0, "5", ["6"])) + "\n")     # duplicate key: last wins
        assert load_done(p) == {("A", 0)}
        q = os.path.join(d, "d.jsonl")
        with open(q, "w") as f:
            for i in range(3):
                f.write(json.dumps({"sample_id": f"S{i}", "question": "q", "steps": ["x"]}) + "\n")
            f.write(json.dumps({"sample_id": "S9", "question": "q", "steps": []}) + "\n")
        assert [s["sample_id"] for s in read_jsonl([q])] == ["S0", "S1", "S2"]
        assert [s["sample_id"] for s in read_jsonl([q], wanted={"S1"})] == ["S1"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
