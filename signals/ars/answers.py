"""Answer extraction and the answer-equivalence rule used by the ARS stability signal.

Equivalence rule (documented design choice for math data; the ARS paper uses an LLM judge
for free-form QA): two answers agree if, after normalisation, the strings are identical or
both are numbers that are equal (fractions, thousands separators and trailing units allowed).
Pure Python, no torch.
"""
from __future__ import annotations
import math
import re


def _match_brace(s: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_answer(text) -> str | None:
    """Short answer string from raw generated text, or None if nothing usable."""
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    i = t.find("\\boxed{")
    if i != -1:
        open_idx = i + len("\\boxed")
        j = _match_brace(t, open_idx)
        if j != -1:
            inner = t[open_idx + 1:j].strip()
            return inner or None
    line = t.split("\n")[0].strip()
    line = re.sub(r"^(so,?\s*)?(the\s+)?(final\s+)?answer\s*(is|:)?\s*", "", line, flags=re.I)
    line = line.strip().rstrip(".").strip()
    return line or None


def normalize_answer(a: str) -> str:
    s = a.strip().lower()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for tok in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$", "^\\circ", "°"):
        s = s.replace(tok, "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)      # 1,000 -> 1000
    return s.rstrip(".")


_NUM = re.compile(r"^(-?\d+(?:\.\d+)?)(?:[a-z]{3,}|%)?$")
_FRAC_TEX = re.compile(r"^(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}$")
_FRAC_SLASH = re.compile(r"^(-?\d+)/(-?\d+)$")


def to_number(s: str) -> float | None:
    m = _NUM.match(s)
    if m:
        return float(m.group(1))
    m = _FRAC_TEX.match(s)
    if m and int(m.group(3)) != 0:
        return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
    m = _FRAC_SLASH.match(s)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    return None


def answers_agree(a, b):
    """True / False, or None when either answer is missing (that trial is invalid)."""
    if a is None or b is None:
        return None
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return None
    if na == nb:
        return True
    xa, xb = to_number(na), to_number(nb)
    if xa is not None and xb is not None:
        return math.isclose(xa, xb, rel_tol=1e-6, abs_tol=1e-9)
    return False
