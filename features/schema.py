"""Single source of truth for the ReTrace feature-matrix schema.

Every script must import from here. Run scripts from the repo root with
`python -m package.module` so these imports resolve.

Conventions
  label     : 0 = correct step, 1 = error/hallucinated step
  step_id   : ZERO-BASED, contiguous from 0 inside each sample_id
  split     : "train" | "val" | "test"   (assigned per problem, see data/make_splits.py)
  problem_id: group key; all trajectories of one problem share one split
  Signals   : higher value = more suspicious (error-like) for all four signals
"""

KEY_COLUMNS = ["sample_id", "step_id"]
SIGNAL_COLUMNS = ["ARS", "Deviation", "Consistency", "Causal"]
TARGET_COLUMN = "label"
GROUP_COLUMN = "problem_id"
SPLIT_COLUMN = "split"
VALID_SPLITS = ["train", "val", "test"]

# Raw matrix produced by merging the signal files (no split info yet)
FEATURE_COLUMNS = KEY_COLUMNS + SIGNAL_COLUMNS + [TARGET_COLUMN]
# Matrix after training/split_data.py
MATRIX_COLUMNS = KEY_COLUMNS + [GROUP_COLUMN, SPLIT_COLUMN] + SIGNAL_COLUMNS + [TARGET_COLUMN]
