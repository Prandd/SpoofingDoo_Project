# Data Pipeline — Build Notes & Design Decisions

This document describes the data pipeline implemented for the CSS 100-sample pilot phase (Tasks 1–3 in `CLAUDE.md`). It covers what was built, why each design choice was made, and how to use the components.

---

## Overview

Three components implement the full pipeline from raw `.wav` files to fold-ready PyTorch datasets:

| Component | Location | Output |
|-----------|----------|--------|
| Prosody extractor | `experiment/scripts/extract_prosody.py` | `data/features/prosody_manifest.json` |
| PyTorch Dataset | `src/datasets/dataset.py` | `CSSDataset` class |
| Fold maker | `experiment/scripts/make_folds.py` | `experiment/protocols/folds/fold_00.json` … `fold_04.json` |

---

## Task 1 — Offline Prosody Extraction

### Script

```bash
python3 experiment/scripts/extract_prosody.py
# defaults: --wav_dir data/raw/wav/CSS  --out data/features/prosody_manifest.json
```

### What is extracted

Three scalar features per utterance, all computed with **parselmouth** (Python/Praat):

| Feature | Praat command | Rationale |
|---------|--------------|-----------|
| `jitter_mean` | `Get jitter (local)` on PointProcess | Captures micro-pitch irregularity. Thai TTS over-smooths pitch transitions → unnaturally low jitter. |
| `shimmer_mean` | `Get shimmer (local)` on PointProcess | Captures micro-amplitude irregularity. Vocoders often suppress natural shimmer. |
| `f0_std` | `To Pitch` → std of voiced frames | Captures F0 contour variability. Thai tones require large, rapid F0 excursions that TTS often flattens. |

**F0 range:** 75–400 Hz covers both male and female Thai speech, including excited-style high F0.

### Output format

`data/features/prosody_manifest.json` — one entry per utterance ID (filename stem):

```json
{
  "Formal_spk1_utt1": {
    "path": "data/raw/wav/CSS/Bona fide/Formal/Formal_spk1_utt1.wav",
    "jitter_mean": 0.0123,
    "shimmer_mean": 0.0891,
    "f0_std": 31.45
  },
  ...
}
```

### Error handling

If jitter or shimmer extraction fails (e.g., utterance too short for PointProcess), the value is set to `0.0` with a warning — the file is not skipped.

---

## Task 2 — PyTorch Dataset (`CSSDataset`)

### Usage

```python
from datasets.dataset import datasets_from_fold

train_ds, test_ds = datasets_from_fold(
    fold_json=Path("experiment/protocols/folds/fold_00.json"),
    manifest_path=Path("data/features/prosody_manifest.json"),
)

waveform, prosody, label, meta = train_ds[0]
# waveform : FloatTensor (64600,)
# prosody  : FloatTensor (3,)   — [jitter_mean, shimmer_mean, f0_std]
# label    : LongTensor scalar  — 0=genuine, 1=spoof
# meta     : dict — utt_id, speaker, style, path
```

### Design decisions

**Fixed length 64600 samples (≈ 4 s at 16 kHz):** Required by AASIST-L's SincNet input. Implemented in `src/datasets/waveform_utils.pad_or_truncate` (tile-then-crop for short utterances; slice for long ones).

**Label and style derived from path, not stored in manifest:** The CSS directory hierarchy is the ground truth (`Bona fide/` → 0, `Spoofed/` → 1; `Formal/Casual/Excited/` → style). This avoids a separate metadata file and eliminates possible label drift.

**Prosody vector is 3-D `[jitter, shimmer, f0_std]`:** Matches the late-fusion MLP input dimension described in `doc/plan_eng.md`. To change dimensions, update `extract_prosody.py` and the model's `prosody_dim` argument in `src/models/aasist_imported.py`.

**Gaussian noise augmentation:** Enabled for training splits via `augment=True` (default in `datasets_from_fold`). Target SNR is 15 dB — strong enough to improve generalisation, mild enough to preserve prosodic features. Augmentation applies only at load time (online), not saved to disk.

**Audio loading priority:** `soundfile` → `librosa` fallback, with automatic mono-conversion and 16 kHz resampling.

### Public API from `src/datasets/__init__.py`

```python
from datasets import CSSDataset, datasets_from_fold, load_manifest, pad_or_truncate
```

---

## Task 3 — Speaker-Disjoint 5-Fold CV (`make_folds.py`)

### Script

```bash
python3 experiment/scripts/make_folds.py
# defaults: --wav_dir data/raw/wav/CSS  --out_dir experiment/protocols/folds
```

### Dataset structure discovered

108 total files:
- **18 genuine** (`Bona fide/`): 6 speakers × 3 utterances × 3 styles
- **90 spoofed** (`Spoofed/`): 5 TTS systems × 6 speakers × 3 utterances × 3 styles

Speaker-to-style mapping (CSS parallel-speaker design):

| Speaker | Style |
|---------|-------|
| spk1, spk2 | Formal |
| spk3, spk4 | Casual |
| spk5, spk6 | Excited |

### Fold strategy

`sklearn.model_selection.GroupKFold(n_splits=5)` with `groups = speaker_id` guarantees no speaker appears in both train and test in any fold.

```
Fold 0: test = [spk1, spk6]  (36 files)  — cross-style test: Formal+Excited
Fold 1: test = [spk5]         (18 files)
Fold 2: test = [spk4]         (18 files)
Fold 3: test = [spk3]         (18 files)
Fold 4: test = [spk2]         (18 files)
```

Fold 0 is the most interesting for Stage 3 (cross-style evaluation) as it tests on two different styles simultaneously.

### Fold JSON schema

```json
{
  "fold": 0,
  "n_folds": 5,
  "train_files": ["Casual_spk3_utt1", ...],
  "test_files":  ["Formal_spk1_utt1", ...],
  "train_speakers": ["spk2", "spk3", "spk4", "spk5"],
  "test_speakers":  ["spk1", "spk6"],
  "train_labels":   {"Casual_spk3_utt1": 0, ...},
  "test_labels":    {"Formal_spk1_utt1": 0, ...},
  "stats": {
    "train_total": 72, "train_genuine": 12, "train_spoof": 60,
    "test_total":  36, "test_genuine":   6, "test_spoof":  30
  }
}
```

### Automated test

`experiment/tests/test_splits.py::test_fold_files_have_no_speaker_overlap_if_present` reads all generated fold JSONs and asserts `train_speakers ∩ test_speakers = ∅`. Runs as part of `pytest`.

---

## Running the full pipeline

```bash
# Step 1: extract prosody (run once; ~2 min for 108 files)
python3 experiment/scripts/extract_prosody.py

# Step 2: generate fold manifests
python3 experiment/scripts/make_folds.py

# Step 3: verify correctness
pytest experiment/tests/ -v
```

Expected output: **4 tests passed**.

---

## Class imbalance note

The CSS pilot is **highly imbalanced**: 5× more spoofed samples than genuine (90 vs 18). Each fold inherits this ratio. Mitigations to apply during training:

- `pos_weight` in `torch.nn.BCEWithLogitsLoss` (set to `n_spoof / n_genuine ≈ 5`)
- Alternatively, oversample genuine utterances in the DataLoader using `WeightedRandomSampler`

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `praat-parselmouth` | Jitter, shimmer, F0 extraction |
| `librosa` | Audio resampling, fallback load |
| `soundfile` | Fast WAV loading (primary) |
| `scikit-learn` | `GroupKFold` for CV splits |
| `torch` | Dataset base class |
| `numpy` | Numerics |

Install: `pip install praat-parselmouth librosa soundfile scikit-learn torch numpy`
