Label 0 = correct, 1 = error or hallucinated, and indexing is zero-based.
Steps after first_error are labelled 1 by team convention, not independently verified. meta.verified_steps marks the annotation-backed part.
answer_label is final-answer correctness, independent of step errors. It is None for PRM800K.
HaluEval is answer-level only (step_labels=None) and can’t be used for first-error localization.
PRM800K is phase 2 only, and the splits are grouped by problem, with ProcessBench not an untouched benchmark.