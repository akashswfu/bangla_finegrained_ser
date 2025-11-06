import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Load existing manifest
df = pd.read_csv("data/manifest.csv")

# Group split: ensure no same speaker overlaps across splits
# and class balance (emotion)
X = df["path"]
y = df["label"]
groups = df["speaker"]

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

splits = list(splitter.split(X, y, groups))
train_idx, valtest_idx = splits[0]

train_df = df.iloc[train_idx]
val_df = df.iloc[valtest_idx].sample(frac=0.5, random_state=42)
test_df = df.iloc[valtest_idx].drop(val_df.index)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat([train_df, val_df, test_df]).sample(frac=1, random_state=42)
final_df.to_csv("data/manifest_split.csv", index=False)

print("✅ Balanced split complete!")
print(final_df["split"].value_counts())
