# analysis/plot_dataset_distribution.py

import csv
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import yaml

CONFIG_PATH = "configs/base.yaml"   # change if needed

def load_emotions_from_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["data"]["emotions"], cfg["data"]["manifest_path"]

def count_by_split_and_label(manifest_path):
    counts = defaultdict(Counter)  # counts[split][label] = n
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"].strip().lower()
            label = row["label"].strip().lower()
            counts[split][label] += 1
    return counts

def plot_bar_for_split(split, emotions, counts, out_dir="analysis"):
    os.makedirs(out_dir, exist_ok=True)
    split_counts = [counts[split].get(e, 0) for e in emotions]

    plt.figure(figsize=(8, 4))
    plt.bar(emotions, split_counts)
    plt.title(f"Utterance Count per Emotion ({split} split)")
    plt.xlabel("Emotion")
    plt.ylabel("Number of utterances")
    plt.grid(axis="y", alpha=0.3)

    out_path = os.path.join(out_dir, f"dataset_distribution_{split}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")

def main():
    emotions, manifest_path = load_emotions_from_config(CONFIG_PATH)
    print("[INFO] Emotions:", emotions)
    print("[INFO] Manifest:", manifest_path)

    counts = count_by_split_and_label(manifest_path)

    for split in ["train", "val", "test"]:
        total = sum(counts[split].values())
        print(f"[SUMMARY] {split}: {total} samples ->", counts[split])
        if total > 0:
            plot_bar_for_split(split, emotions, counts)

if __name__ == "__main__":
    main()
