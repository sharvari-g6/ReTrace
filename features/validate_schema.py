import pandas as pd

EXPECTED_COLUMNS = [
    "sample_id",
    "step_id",
    "ARS",
    "Deviation",
    "Consistency",
    "Causal",
    "label",
]

df = pd.read_csv("training/example_matrix.csv")

assert list(df.columns) == EXPECTED_COLUMNS

assert df["label"].isin([0, 1]).all()

for column in [
    "ARS",
    "Deviation",
    "Consistency",
    "Causal"
]:
    assert df[column].notna().all()

print("Schema validation successful.")
print("Rows:", len(df))
print("Columns:", len(df.columns))