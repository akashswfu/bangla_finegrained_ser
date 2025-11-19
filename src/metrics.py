# src/metrics.py

from typing import List
import numpy as np

def _safe_macro_f1(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    """
    Macro-F1 with zero_division=0 semantics (no NaN even if a class is missing).
    """
    if len(y_true) == 0:
        return 0.0

    # build confusion counts per class (TP, FP, FN)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = (precision + recall)
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
        f1s.append(f1)

    return float(np.mean(f1s)) if len(f1s) > 0 else 0.0


def _safe_uar(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    """
    Unweighted Average Recall (macro recall) with zero_division=0 semantics.
    """
    if len(y_true) == 0:
        return 0.0

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    recalls = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)

    return float(np.mean(recalls)) if len(recalls) > 0 else 0.0


# keep these names to match your train.py imports
def macro_f1(y_true: List[int], y_pred: List[int], num_classes: int = None) -> float:
    if num_classes is None:
        # infer from data (safe)
        k = 0
        if len(y_true) > 0: k = max(k, int(max(y_true)))
        if len(y_pred) > 0: k = max(k, int(max(y_pred)))
        num_classes = k + 1
    return _safe_macro_f1(y_true, y_pred, num_classes)


def uar(y_true: List[int], y_pred: List[int], num_classes: int = None) -> float:
    if num_classes is None:
        k = 0
        if len(y_true) > 0: k = max(k, int(max(y_true)))
        if len(y_pred) > 0: k = max(k, int(max(y_pred)))
        num_classes = k + 1
    return _safe_uar(y_true, y_pred, num_classes)
