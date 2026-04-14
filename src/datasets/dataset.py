"""CSS pilot Dataset — raw waveform (64600 samples) + 3-D prosody vector.

Typical usage
-------------
    from datasets.dataset import CSSDataset, datasets_from_fold, load_manifest

    manifest = load_manifest(Path("data/features/prosody_manifest.json"))
    train_ds, test_ds = datasets_from_fold(
        fold_json=Path("experiment/protocols/folds/fold_00.json"),
        manifest_path=Path("data/features/prosody_manifest.json"),
    )

Each __getitem__ returns:
    waveform  : torch.FloatTensor of shape (64600,)
    prosody   : torch.FloatTensor of shape (3,)   — [jitter_mean, shimmer_mean, f0_std]
    label     : torch.LongTensor scalar            — 0 = genuine, 1 = spoof
    meta      : dict — {'utt_id', 'speaker', 'style', 'path'}
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from datasets.waveform_utils import pad_or_truncate

_ROOT = Path(__file__).resolve().parents[2]
_MAX_LEN: int = 64600
_LABEL_MAP: Dict[str, int] = {"Bona fide": 0, "Spoofed": 1}
_STYLES = frozenset(("Formal", "Casual", "Excited"))


# ---------------------------------------------------------------------------
# Path-based helpers
# ---------------------------------------------------------------------------

def _parse_speaker(stem: str) -> str:
    """Return the speaker token (e.g. 'spk1') embedded in a CSS filename stem.

    CSS naming:
      Bona fide → '{Style}_{spkX}_{uttY}'
      Spoofed   → '{TTSModel}_{Style}_{spkX}_{uttY}'
    The speaker token is always of the form 'spkN'.
    """
    m = re.search(r"(spk\d+)", stem)
    return m.group(1) if m else "unknown"


def _infer_label(wav_path: Path) -> int:
    """Derive 0/1 label from the CSS directory hierarchy."""
    for part in wav_path.parts:
        if part in _LABEL_MAP:
            return _LABEL_MAP[part]
    raise ValueError(
        f"Cannot infer label from path '{wav_path}'. "
        "Expected 'Bona fide' or 'Spoofed' in the path."
    )


def _infer_style(wav_path: Path) -> str:
    """Derive speech style (Formal/Casual/Excited) from the path."""
    for part in wav_path.parts:
        if part in _STYLES:
            return part
    return "Unknown"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CSSDataset(Dataset):
    """PyTorch Dataset for the CSS (Chula Spoofed Speech) pilot corpus.

    Parameters
    ----------
    manifest : dict
        Loaded prosody_manifest.json (utt_id → {path, jitter_mean, shimmer_mean, f0_std}).
    file_list : list of str
        Utterance IDs (filename stems) for this split.
    root : Path
        Project root — absolute paths are built as root / entry["path"].
    max_len : int
        Fixed waveform length in samples (default 64600 ≈ 4 s at 16 kHz).
    augment : bool
        If True, add Gaussian white noise during __getitem__ (use for training only).
    aug_snr_db : float
        Target SNR in dB for Gaussian noise augmentation.
    """

    def __init__(
        self,
        manifest: Dict,
        file_list: List[str],
        root: Path = _ROOT,
        max_len: int = _MAX_LEN,
        augment: bool = False,
        aug_snr_db: float = 15.0,
    ) -> None:
        missing = [uid for uid in file_list if uid not in manifest]
        if missing:
            raise KeyError(
                f"Utterance IDs not found in manifest: {missing[:5]}"
                + (" (and more)" if len(missing) > 5 else "")
            )

        self.manifest = manifest
        self.file_list = list(file_list)
        self.root = Path(root)
        self.max_len = max_len
        self.augment = augment
        self.aug_snr_db = aug_snr_db

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tensor, Tensor, Tensor, dict]:
        utt_id = self.file_list[idx]
        entry = self.manifest[utt_id]

        # ---- waveform --------------------------------------------------
        wav_path = self.root / entry["path"]
        waveform = self._load_wav(wav_path)

        if self.augment:
            waveform = self._add_noise(waveform)

        waveform_tensor = torch.from_numpy(waveform).float()  # (64600,)

        # ---- prosody vector (3-D) --------------------------------------
        prosody = np.array(
            [entry["jitter_mean"], entry["shimmer_mean"], entry["f0_std"]],
            dtype=np.float32,
        )
        prosody_tensor = torch.from_numpy(prosody)  # (3,)

        # ---- label -----------------------------------------------------
        label_val = _infer_label(wav_path)
        label_tensor = torch.tensor(label_val, dtype=torch.long)

        meta = {
            "utt_id": utt_id,
            "speaker": _parse_speaker(utt_id),
            "style": _infer_style(wav_path),
            "path": str(wav_path),
        }

        return waveform_tensor, prosody_tensor, label_tensor, meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_wav(self, path: Path) -> np.ndarray:
        """Load waveform as mono float32 at 16 kHz, fixed to max_len samples."""
        try:
            import soundfile as sf
            x, sr = sf.read(str(path), dtype="float32")
        except Exception:
            import librosa
            x, sr = librosa.load(str(path), sr=None, mono=True)
            x = x.astype(np.float32)

        if x.ndim > 1:
            x = x.mean(axis=1)

        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000).astype(np.float32)

        return pad_or_truncate(x, self.max_len)

    def _add_noise(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian white noise at self.aug_snr_db dB SNR."""
        signal_power = float(np.mean(x ** 2))
        if signal_power < 1e-10:
            return x
        noise_power = signal_power / (10.0 ** (self.aug_snr_db / 10.0))
        noise = np.random.randn(len(x)).astype(np.float32) * float(np.sqrt(noise_power))
        return x + noise


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> Dict:
    """Load prosody_manifest.json into a dict."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def datasets_from_fold(
    fold_json: Path,
    manifest_path: Path,
    root: Path = _ROOT,
    augment_train: bool = True,
    aug_snr_db: float = 15.0,
) -> Tuple[CSSDataset, CSSDataset]:
    """Return (train_dataset, test_dataset) for one fold JSON file.

    Parameters
    ----------
    fold_json : Path
        Path to fold_XX.json generated by make_folds.py.
    manifest_path : Path
        Path to prosody_manifest.json generated by extract_prosody.py.
    root : Path
        Project root directory (default: auto-detected from __file__).
    augment_train : bool
        Apply noise augmentation to the training split.
    aug_snr_db : float
        SNR for noise augmentation.
    """
    with open(fold_json, "r", encoding="utf-8") as f:
        fold = json.load(f)

    manifest = load_manifest(manifest_path)

    train_ds = CSSDataset(
        manifest,
        fold["train_files"],
        root=root,
        augment=augment_train,
        aug_snr_db=aug_snr_db,
    )
    test_ds = CSSDataset(
        manifest,
        fold["test_files"],
        root=root,
        augment=False,
    )
    return train_ds, test_ds
