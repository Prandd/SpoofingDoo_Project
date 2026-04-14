# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Thai speech spoofing detection project using AASIST-L model with CSS (Chula Spoofed Speech) dataset. The project focuses on adapting AASIST-L (pre-trained on English ASVspoof) for Thai speech spoofing detection, with emphasis on exploiting prosodic cues (tone and intonation patterns) unique to Thai language.

## Development Commands

### Installation
```bash
cd SpoofingDoo_Project
pip install -e ".[dev]"
```

### Testing
```bash
pytest                    # Run all tests
pytest experiment/tests   # Run only experiment tests
```

### Running Examples
```bash
python experiment/scripts/single_attention.py     # Attention visualization
python experiment/scripts/compare_attention.py    # Compare attention maps
python -m features.pitch_extraction               # Pitch demo (from repo root with PYTHONPATH=src)
```

### Running with PYTHONPATH (alternative to editable install)
```bash
PYTHONPATH=src python -m features.pitch_extraction
```

## Architecture Overview

The project follows a clean separation between reusable library code and experiment-specific code:

### Core Architecture Components

- **AASIST-L Model**: Lightweight (85k parameters) spoofing detection model using:
  - SincNet layer for raw waveform feature extraction
  - 2D feature maps through residual blocks
  - Temporal and spectral graph construction
  - Heterogeneous Stacking Graph Attention Layer (HS-GAL)
  - Max Graph Operation (MGO) for aggregation

- **Transfer Learning Strategy**: Load AASIST-L pre-trained on English ASVspoof 2019, then adapt for Thai domain with minimal modifications to avoid tensor shape bugs

- **Thai-Specific Features**: Late fusion of prosodic features (jitter, shimmer, F0 statistics) extracted offline using parselmouth/librosa

### Directory Structure

```
SpoofingDoo_Project/
├── src/                          # Reusable library code
│   ├── models/                   # AASIST-L wrapper, prosody MLP
│   ├── datasets/                 # PyTorch datasets, CV splits
│   ├── features/                 # Offline feature extraction
│   ├── train/                    # Training loops, augmentation
│   └── eval/                     # EER, AUC, ROC/DET evaluation
├── experiment/                   # ALL experiment artifacts
│   ├── configs/                  # Stage configs (YAML)
│   ├── protocols/                # 5-fold speaker-disjoint CV splits
│   ├── scripts/                  # CLI entrypoints
│   ├── tests/                    # pytest for experiment code
│   ├── runs/                     # Output directory (gitignored)
│   └── notebooks/                # Analysis and visualization
├── data/                         # Local audio, features (gitignored)
└── doc/                          # Plans and architecture docs
```

### Development Stages

The project is structured around a 4-stage development approach:

1. **Stage 0**: Zero-shot baseline (pre-trained AASIST-L on Thai samples)
2. **Stage 1**: Fine-tune readout only (freeze backbone)
3. **Stage 2**: Add prosody late fusion (jitter/shimmer + F0 stats)
4. **Stage 3**: Cross-style evaluation (formal vs excited speech)

Each stage has corresponding config files in `experiment/configs/`.

## Key Constraints and Design Decisions

- **Small Dataset**: Only ~100 pilot samples, requires careful overfitting prevention
- **Transfer Learning**: Always use pre-trained weights, freeze most layers
- **Speaker-Disjoint Splits**: 5-fold CV with no speaker overlap between train/test
- **Offline Feature Extraction**: Prosodic features cached as .npy files for faster training
- **Conservative Modifications**: Avoid changing graph construction/HS-GAL/MGO to prevent tensor shape bugs

## Testing Strategy

- **Unit Tests**: `experiment/tests/` covers splits validation, dataset shapes, metrics
- **Smoke Tests**: Quick integration runs using `experiment/configs/smoke_*.yaml`
- **Cross-Validation**: Speaker-disjoint 5-fold protocol in `experiment/protocols/`

## Import Structure

The project uses setuptools with src layout. After `pip install -e .`, import as:
```python
from models import Model                    # AASIST-L wrapper
from features.pitch_extraction import ...  # Prosody extraction
from datasets import ...                   # PyTorch datasets
from train import ...                      # Training utilities
from eval import ...                       # Evaluation metrics
```

## Key Files to Understand

- `src/models/aasist_imported.py`: Core AASIST-L model with optional prosody fusion
- `experiment/configs/`: YAML configs for each development stage
- `experiment/protocols/README.md`: CV split requirements and speaker disjoint policy
- `doc/plan_eng.md`: Detailed research roadmap and technical rationale
- `doc/structure.md`: Directory organization and experiment framework

## 🚨 CRITICAL CONSTRAINTS (READ ALWAYS)
- **Data Limitation:** We ONLY have ~100 pilot samples from the CSS dataset.
- **NO Training from Scratch:** Do NOT build deep models from scratch. We will OVERFIT.
- **Rule:** ALWAYS use Transfer Learning. Load ASVspoof-pretrained AASIST-L, **FREEZE the SincNet and Graph backbone (`requires_grad=False`)**, and ONLY fine-tune the final readout / MLP fusion layer.
- **Audio Shapes:** Waveforms must be cropped/padded to exactly 64,600 samples.

## 📌 CURRENT PROGRESS (Update this manually when starting a new session)
- [x] Structure defined.
- [x] Task 1: Offline Prosody Extraction.
- [x] Task 2: PyTorch Dataset & 5-fold CV splits.
- [x] Task 3: AASIST-L Late Fusion MLP Modification. (`AASISTWithProsody` in src/models/aasist_imported.py) **← BASELINE MODEL**
- [x] Task 4: Training Loop & Augmentation. (`experiment/scripts/baseline_train.py`; 4 figures → experiment/runs/figures/) **← BASELINE SCRIPT**

---

## 🚀 PHASE 2 / CONTINGENCY PLAN: 10% CSS DATASET SCALE
**Trigger:** Activate this plan ONLY IF the team successfully acquires the 133,210 samples (~160 hours) from the CSS dataset. Ignore this section during the 100-sample pilot phase.

**1. Dataset & Split Strategy:**
- Abandon 5-Fold CV. Switch to a **Train (70%) / Val (15%) / Test (15%)** split.
- Enforce strict Speaker-Disjoint splitting (Test speakers must never appear in Train).

**2. Optimized Data Pipeline:**
- Implement parallel offline feature extraction using `joblib` or `torch.multiprocessing`.
- Dataloader must be optimized for I/O: Set `num_workers` to CPU core count and enable `pin_memory=True`.

**3. ML Modeling Updates (Full Fine-Tuning):**
- **UNFREEZE** the entire AASIST-L architecture (`requires_grad=True`).
- Use **Discriminative Learning Rates**: e.g., `1e-6` for SincNet/Graph backbone, and `1e-4` for the readout head.
- Implement Cosine Annealing LR scheduler.

**4. Infrastructure (Cloud Computing):**
- Local training is no longer viable. Deploy training scripts via Docker to a Cloud GPU.
- **Recommended GPU:** NVIDIA A10/A10G (24GB VRAM). 
- **Providers:** AWS `g5.xlarge` (~$1.00/hr) or Lambda Labs 1x A10 (~$0.60/hr). Focus on fast NVMe SSD storage to prevent I/O bottlenecks.