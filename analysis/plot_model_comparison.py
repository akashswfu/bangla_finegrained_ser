# analysis/plot_model_comparison.py

import os
import matplotlib.pyplot as plt
import numpy as np

# === EDIT HERE if you retrain and get new scores ===
baseline_val_f1  = 0.7344
baseline_val_uar = 0.7483
baseline_test_f1  = 0.7238
baseline_test_uar = 0.7330

weakseg_val_f1  = 0.7322
weakseg_val_uar = 0.7364
weakseg_test_f1  = 0.7389
weakseg_test_uar = 0.7417
# ===================================================

def plot_metric(metric_name, baseline_val, baseline_test, weakseg_val, weakseg_test, out_path):
    labels = ["Baseline", "WeakSeg"]
    val_scores  = [baseline_val, weakseg_val]
    test_scores = [baseline_test, weakseg_test]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, val_scores,  width, label="Val")
    plt.bar(x + width/2, test_scores, width, label="Test")

    plt.xticks(x, labels)
    plt.ylabel(metric_name)
    plt.ylim(0.0, 1.0)
    plt.title(f"{metric_name} Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def main():
    plot_metric(
        "Macro-F1",
        baseline_val_f1,
        baseline_test_f1,
        weakseg_val_f1,
        weakseg_test_f1,
        "analysis/model_comparison_macro_f1.png"
    )

    plot_metric(
        "UAR",
        baseline_val_uar,
        baseline_test_uar,
        weakseg_val_uar,
        weakseg_test_uar,
        "analysis/model_comparison_uar.png"
    )


if __name__ == "__main__":
    main()
