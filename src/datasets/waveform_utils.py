"""Length normalization for raw waveforms (no heavy audio deps)."""

import numpy as np


def pad_or_truncate(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    """Ensures audio is exactly `max_len` samples (AASIST-style fixed length)."""
    x = np.asarray(x)
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats,))[:max_len]
    return padded_x
