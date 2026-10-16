"""Where each reasoning step sits in the prompt (characters, then tokens). Pure Python."""
from __future__ import annotations
from signals.ars.prompts import build_prefix


def step_char_spans(question: str, steps: list) -> list:
    """Character span [a, b) of each step block ("Step i: text\\n") inside build_prefix(question, steps, n-1)."""
    spans, prev = [], len(build_prefix(question, steps, -1))      # header only
    for t in range(len(steps)):
        end = len(build_prefix(question, steps, t))
        spans.append((prev, end))
        prev = end
    return spans


def token_indices(offsets, a: int, b: int) -> list:
    """Indices of tokens whose first character lies in [a, b); offsets = [(start, end), ...]."""
    return [i for i, (s, e) in enumerate(offsets) if e > s and a <= s < b]
