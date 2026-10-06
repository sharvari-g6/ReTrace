import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("training/example_matrix.csv")

# Get unique reasoning trajectories
sample_ids = df["sample_id"].unique()

# 70% train, 30% temporary
train_ids, temp_ids = train_test_split(
    sample_ids,
    test_size=0.30,
    random_state=42
)

# 15% validation, 15% test
val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.50,
    random_state=42
)

train_df = df[df["sample_id"].isin(train_ids)]
val_df = df[df["sample_id"].isin(val_ids)]
test_df = df[df["sample_id"].isin(test_ids)]

print("Train samples:", len(train_ids))
print("Validation samples:", len(val_ids))
print("Test samples:", len(test_ids))