"""Dataset helpers — waveform length / prosody alignment (extend when Dataset lands)."""

import numpy as np

from datasets.waveform_utils import pad_or_truncate


def test_pad_or_truncate_fixed_length():
    x = np.ones(1000, dtype=np.float32)
    out = pad_or_truncate(x, max_len=64600)
    assert out.shape == (64600,)
    long = np.ones(100000, dtype=np.float32)
    out_long = pad_or_truncate(long, max_len=64600)
    assert out_long.shape == (64600,)
    assert np.allclose(out_long, long[:64600])
