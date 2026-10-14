import pandas as pd

MATRIX_PATH = "training/example_matrix.csv"
SPLITS_PATH = "data/processed/sample_splits.csv"   # produced by: python -m data.make_splits

df = pd.read_csv(MATRIX_PATH)
splits = pd.read_csv(SPLITS_PATH)  # sample_id, source, problem_id, split

# Splits are defined at PROBLEM-GROUP level (see data/splits/split_manifest.json),
# not by sample_id. Every sample inherits the split of its problem.
df = df.merge(
    splits[["sample_id", "problem_id", "split"]],
    on="sample_id",
    how="left",
    validate="many_to_one",
)

missing = df.loc[df["split"].isna(), "sample_id"].unique()
assert len(missing) == 0, f"sample_ids missing from split manifest: {list(missing[:5])}"

# Safety check: no problem may appear in more than one split
assert (df.groupby("problem_id")["split"].nunique() == 1).all(), "problem leaked across splits"

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]

for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    print(f"{name}: {part['sample_id'].nunique()} trajectories, "
          f"{part['problem_id'].nunique()} unique problems, {len(part)} step rows")