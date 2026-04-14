"""EER / AUC helpers — extend when eval module is implemented."""

import numpy as np


def test_perfect_scores_imply_zero_eer_hand_tuned():
    # Placeholder: real EER test once `eval` exposes a function.
    scores = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    assert scores.shape == labels.shape
