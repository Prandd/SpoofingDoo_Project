# Environment Setup – Pilot Phase (100 samples)

**Date:** 2026-04-14  
**Scope:** Local development setup for the SpoofingDoo project, 100-sample CSS pilot phase only.

---

## What Was Done

Created `requirements.txt` in the project root with the minimal set of packages needed to run the full pilot pipeline end-to-end, without pulling in heavy extras that slow down local setup.

---

## Package Rationale

| Package | Version | Why needed |
|---|---|---|
| `torch` / `torchaudio` | ≥2.0 | Core model training; torchaudio handles WAV loading and resampling natively |
| `praat-parselmouth` | ≥0.4.3 | Extracts jitter, shimmer, and F0 statistics offline (Task 1) |
| `librosa` | ≥0.10.0 | Supplementary audio loading, resampling, STFT; also used in feature scripts |
| `soundfile` | ≥0.12.1 | Fast WAV backend shared by librosa and torchaudio |
| `scikit-learn` | ≥1.3 | `StratifiedKFold` / `GroupKFold` for 5-fold speaker-disjoint CV splits (Task 2) |
| `numpy` | ≥1.24 | Core numerical array ops throughout |
| `scipy` | ≥1.11 | Signal utilities; already a transitive dep of librosa |
| `matplotlib` | ≥3.7 | ROC / DET curves and attention-map visualisation |
| `pyyaml` | ≥6.0 | Loads `experiment/configs/*.yaml` stage configs |
| `pytest` | ≥7.0 | Unit and smoke tests under `experiment/tests/` |

---

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install PyTorch for your hardware

PyTorch must be installed **before** the rest so pip picks the right binary.

```bash
# CPU-only (safe default)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (MPS acceleration)
pip install torch torchaudio

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the project package in editable mode

```bash
cd SpoofingDoo_Project
pip install -e ".[dev]"
```

### 5. Verify the setup

```bash
python - <<'EOF'
import torch, torchaudio, parselmouth, librosa, sklearn, matplotlib
print("torch     :", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("parselmouth:", parselmouth.__version__)
print("librosa   :", librosa.__version__)
print("sklearn   :", sklearn.__version__)
print("All imports OK")
EOF
```

Expected output (versions will vary):

```
torch     : 2.x.x
torchaudio: 2.x.x
parselmouth: 0.4.x
librosa   : 0.10.x
sklearn   : 1.x.x
All imports OK
```

---

## Deliberately Excluded Packages

These were left out to keep setup fast and lightweight for the 100-sample pilot:

| Package | Reason excluded | Add when |
|---|---|---|
| `torch-audiomentations` | Heavy; downloads audio IR datasets on first run | Stage 1 fine-tuning (augmentation) |
| `jupyter` / `ipykernel` | Notebooks are optional for core pipeline | Local analysis only; `pip install jupyter` ad-hoc |
| `tensorboard` | Overkill for 100 samples; use `print` logs | Phase 2 full-scale training |
| `pandas` | Not needed; `.npy` and `dict` are sufficient | Only if adding tabular metadata |

---

## Key Constraints Reminder

- Waveforms must be cropped / padded to exactly **64,600 samples** before entering AASIST-L.
- Always load pre-trained ASVspoof weights; **never train from scratch**.
- Freeze `SincNet` and graph backbone (`requires_grad=False`); only fine-tune readout / MLP fusion.
- Prosodic features (jitter, shimmer, F0 stats) are extracted **offline** and cached as `.npy` files.

---

## Next Steps (Task Order from `plan_eng.md`)

1. **Task 1** – `src/features/pitch_extraction.py`: offline prosody extraction with `parselmouth` + `librosa`.
2. **Task 2** – `src/datasets/`: PyTorch `Dataset` + 5-fold speaker-disjoint CV splits with `scikit-learn`.
3. **Task 3** – `src/models/aasist_imported.py`: add late-fusion MLP branch for prosody vector.
4. **Task 4** – `src/train/`: training loop with Gaussian noise augmentation, early stopping, EER + AUC.
