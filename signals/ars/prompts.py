"""Boundary prompt for ARS. Same text format as the hidden-state extraction notebook:
    Question: {q}\\nReasoning:\\nStep 1: ...\\nStep 2: ...\\n
so the prefix tokens match the extracted states. The answer cue is appended after step t.
"""
ANSWER_CUE = "Based only on the steps above, the final answer is:"


def build_prefix(question: str, steps: list, t: int) -> str:
    """Question + steps[0..t] (inclusive)."""
    body = "".join(f"Step {i + 1}: {s}\n" for i, s in enumerate(steps[: t + 1]))
    return f"Question: {question}\nReasoning:\n{body}"


def build_prompt(question: str, steps: list, t: int) -> str:
    return build_prefix(question, steps, t) + ANSWER_CUE
