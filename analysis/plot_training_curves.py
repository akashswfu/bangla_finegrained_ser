# analysis/plot_training_curves.py

import os
import csv
import matplotlib.pyplot as plt


def load_log(path):
    epochs, train_loss, val_f1, val_uar = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_f1.append(float(row["val_macro_f1"]))
            val_uar.append(float(row["val_uar"]))
    return epochs, train_loss, val_f1, val_uar


def plot_curves(epochs, train_loss, val_f1, val_uar, title_prefix, out_dir="analysis"):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Train loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title(f"{title_prefix} - Train Loss")
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_train_loss.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")

    # 2) Val Macro-F1 and UAR
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_f1, marker="o", label="Val Macro-F1")
    plt.plot(epochs, val_uar, marker="s", label="Val UAR")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.title(f"{title_prefix} - Validation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_val_metrics.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def main():
    base_log = "outputs/logs/baseline_training.csv"
    weak_log = "outputs/logs/weakseg_training.csv"

    if os.path.exists(base_log):
        ep, tr, f1, uar = load_log(base_log)
        plot_curves(ep, tr, f1, uar, "Baseline")
    else:
        print(f"[WARN] {base_log} not found, skipping baseline curves")

    if os.path.exists(weak_log):
        ep, tr, f1, uar = load_log(weak_log)
        plot_curves(ep, tr, f1, uar, "WeakSeg")
    else:
        print(f"[WARN] {weak_log} not found, skipping weakseg curves")


if __name__ == "__main__":
    main()
