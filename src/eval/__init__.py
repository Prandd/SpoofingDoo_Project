"""EER, AUC, and ROC evaluation helpers for anti-spoofing."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and its threshold.

    Parameters
    ----------
    labels : np.ndarray, shape (N,)
        Ground-truth binary labels. 1 = spoof (positive), 0 = genuine (negative).
    scores : np.ndarray, shape (N,)
        Model output probabilities or logits for the positive class (higher →
        more likely spoof).

    Returns
    -------
    eer : float
        Equal error rate in [0, 1].
    threshold : float
        Decision threshold at EER point.
    """
    # Sort by descending score
    sorted_idx = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_idx]

    n_pos = int(labels.sum())          # total spoof samples
    n_neg = len(labels) - n_pos        # total genuine samples

    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0

    # Sweep thresholds (one per unique score)
    tp = 0
    fp = 0
    eer = 1.0
    eer_threshold = scores[sorted_idx[0]]

    for i, lbl in enumerate(sorted_labels):
        if lbl == 1:
            tp += 1
        else:
            fp += 1

        fnr = 1.0 - tp / n_pos   # false negative rate (miss rate)
        fpr = fp / n_neg          # false positive rate

        if fpr >= fnr:
            # Linear interpolation between current and previous step
            if i == 0:
                eer = (fnr + fpr) / 2.0
            else:
                prev_lbl = sorted_labels[i - 1]
                prev_tp = tp - int(prev_lbl == 1)
                prev_fp = fp - int(prev_lbl == 0)
                prev_fnr = 1.0 - prev_tp / n_pos
                prev_fpr = prev_fp / n_neg
                # Interpolate where fnr == fpr
                denom = (fpr - prev_fpr) - (fnr - prev_fnr)
                if abs(denom) < 1e-12:
                    eer = (fnr + fpr) / 2.0
                else:
                    t = (prev_fnr - prev_fpr) / denom
                    eer = prev_fnr + t * (fnr - prev_fnr)
            eer_threshold = float(scores[sorted_idx[i]])
            break

    return float(eer), float(eer_threshold)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC score.

    Parameters
    ----------
    labels : np.ndarray
        Binary ground-truth labels (1 = spoof, 0 = genuine).
    scores : np.ndarray
        Model scores (higher → more likely spoof).

    Returns
    -------
    auc : float
        Area under ROC curve, in [0, 1].
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        # Manual trapezoidal AUC fallback
        sorted_idx = np.argsort(scores)[::-1]
        sorted_labels = labels[sorted_idx]
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.0
        tp = fp = 0
        auc = 0.0
        prev_fp = 0
        for lbl in sorted_labels:
            if lbl == 1:
                tp += 1
            else:
                fp += 1
                auc += tp
        return float(auc) / (n_pos * n_neg)
