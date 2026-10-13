import hashlib
import re
import unicodedata


def normalize_problem(text: str) -> str:
    """Canonical form used for grouping and cross-dataset dedup."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def problem_hash(text: str) -> str:
    return hashlib.sha1(normalize_problem(text).encode("utf-8")).hexdigest()[:16]

"""Unified trajectory format shared by all loaders (Member B).

Conventions (match Member A's schema):
  label 0 = correct, 1 = error/hallucinated
  first_error and step_labels are 0-indexed (step_id in the feature matrix is 1-indexed)
"""

CORRECT = 0
ERROR = 1
# Steps AFTER the first error are not annotated by the datasets.
# By team convention (example_matrix.csv) they are labelled ERROR.
AFTER_ERROR_LABEL = ERROR

REQUIRED_KEYS = [
    "sample_id", "source", "problem_id", "question", "steps",
    "step_labels", "first_error", "answer", "answer_label", "meta",
]


def make_record(*, sample_id, source, question, steps, step_labels,
                first_error, answer, answer_label, meta=None):
    rec = {
        "sample_id": sample_id,
        "source": source,
        "problem_id": problem_hash(question),
        "question": question,
        "steps": steps,
        "step_labels": step_labels,
        "first_error": first_error,
        "answer": answer,
        "answer_label": answer_label,
        "meta": meta or {},
    }
    validate_record(rec)
    return rec


def validate_record(rec):
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    assert not missing, f"missing keys: {missing}"

    steps = rec["steps"]
    assert isinstance(steps, list) and len(steps) > 0, "steps must be a non-empty list"

    labels = rec["step_labels"]
    if labels is not None:
        assert len(labels) == len(steps), "step_labels length != steps length"
        assert all(l in (CORRECT, ERROR) for l in labels), "step_labels must be 0/1"
        fe = rec["first_error"]
        if fe is None:
            assert all(l == CORRECT for l in labels), "no first_error but an error label exists"
        else:
            assert 0 <= fe < len(steps), "first_error out of range"
            assert all(l == CORRECT for l in labels[:fe]), "error label before first_error"
            assert labels[fe] == ERROR, "step at first_error is not labelled error"

    assert rec["answer_label"] in (CORRECT, ERROR, None), "answer_label must be 0/1/None"