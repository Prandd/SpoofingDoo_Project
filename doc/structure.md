# SpoofingDoo — project framework and directory structure

This document aligns the codebase with the roadmap in `doc/plan_eng.md`: **AASIST-L** (ASVspoof-pretrained) on **CSS** pilot data (~100 samples), optional **late-fusion prosody** (jitter, shimmer, \(F_0\) stats via `parselmouth`), **5-fold speaker-disjoint CV**, and metrics **EER** / **AUC**.

**Rule:** All **experiment** artifacts (protocols, splits, runs, logs, checkpoints, notebooks, evaluation scripts, and **automated tests** for experiment code) live under **`experiment/`**. Library-style modules that are reused across runs may live at repo root or under `src/` as you prefer; this file recommends a clean split.

---

## 1. High-level framework

| Layer | Role |
|--------|------|
| **Data** | Raw CSS `.wav`, metadata (speaker ID, style), cached offline prosody vectors (`.npy` / `.json`). |
| **Features** | Offline extraction (parselmouth / librosa); no training logic. |
| **Models** | AASIST-L wrapper, optional prosody MLP, backup ResNet18+LFCC (emergency). |
| **Training** | Fold loops, freezing policy, augmentation, early stopping, optimiser. |
| **Evaluation** | EER, AUC, ROC/DET export; cross-style analysis hooks. |
| **Experiments** | Everything runnable as a “study”: configs, seeds, outputs — **only under `experiment/`**. |

**Stages (from plan):** Stage 0 zero-shot baseline → Stage 1 frozen-backbone readout fine-tune → Stage 2 prosody late fusion → Stage 3 cross-style evaluation. Each stage maps to one or more **experiment configs** under `experiment/`.

---

## 2. Recommended directory tree

```text
SpoofingDoo_Project/
├── doc/                          # Plans, structure, notes (this file, plan_eng.md, …)
├── data/                         # Gitignored: raw CSS audio + tables (not committed)
│   ├── raw/wav/
│   ├── metadata/                 # speaker_id, label, style, file_id, …
│   └── features/                 # utterance-level prosody caches (*.npy, manifest.json)
├── src/                          # importable packages (setuptools: `pip install -e .`)
│   ├── datasets/
│   ├── models/
│   ├── features/                 # pitch, mel comparison, zip → wav extraction
│   ├── train/
│   └── eval/
├── experiment/                   # ALL experiments and tests live here
│   ├── configs/                  # YAML/JSON: stage, folds, lr, freeze flags, paths
│   ├── protocols/              # 5-fold CV: speaker-disjoint split definitions
│   │   ├── folds/                # e.g. fold_00.json … fold_04.json (train/test file lists)
│   │   └── README.md             # how splits were built (no leakage)
│   ├── scripts/                  # CLI entrypoints: extract_features, train_fold, eval_fold
│   ├── runs/                     # Gitignored: per-run outputs (see below)
│   ├── notebooks/                # EDA, error analysis, tone-level plots
│   ├── tests/                    # pytest (or unittest): ONLY code used by experiments
│   │   ├── test_splits.py        # no speaker overlap across train/test
│   │   ├── test_dataset.py       # shapes, pad/crop 64600, prosody vector alignment
│   │   └── test_metrics.py       # EER/AUC sanity on toy scores
│   └── ablations/                # Optional: minimal table from plan (baseline vs fusion; cross-style)
│       ├── baseline_vs_prosody/
│       └── cross_style_formal_train_excited_spoof/
└── README.md
```

**`experiment/runs/`** (typical layout, gitignored):

```text
experiment/runs/<run_id>/
├── config.resolved.yaml          # copy of config + git hash + seed
├── logs/                         # tensorboard / csv
├── checkpoints/
├── metrics/                      # per-fold EER, AUC, ROC points
└── figures/                      # ROC/DET for slides
```

---

## 3. Mapping plan_eng.md → folders

| Plan item | Location |
|-----------|----------|
| Offline prosody (jitter, shimmer, \(F_0\) stats) | `src/` feature module + `experiment/scripts/extract_prosody.py`; outputs under `data/features/` |
| PyTorch `Dataset` + 5-fold CV | `src/datasets/` + fold JSON under `experiment/protocols/folds/` |
| AASIST-L + prosody `forward` | `src/models/aasist_imported.py` (`from models import Model`) |
| Training loop, freeze, augmentation, early stopping | `src/train/` + `experiment/scripts/train.py` |
| Stage 0–3 | `experiment/configs/stage0_zero_shot.yaml`, `stage1_finetune_readout.yaml`, … |
| Ablations (baseline vs prosody; cross-style) | `experiment/ablations/` + configs pointing at same `protocols/` |
| Deliverables (ROC, EER table, error analysis) | Generated under `experiment/runs/.../figures/` and `notebooks/` |

---

## 4. Tests and “test structure” under `experiment/`

- **Automated tests:** `experiment/tests/` — run with `pytest experiment/tests` from repo root (or set `pythonpath` to include `src`).
- **Held-out evaluation / folds:** not named `test/` at repo root; **split manifests** live in `experiment/protocols/folds/`.
- **Smoke / integration runs:** short configs in `experiment/configs/smoke_*.yaml` writing to `experiment/runs/_smoke/`.

This keeps **all** experiment-related quality gates and evaluation protocol **scoped under `experiment/`**, as required.

---

## 5. Naming and conventions

- **Speaker-disjoint folds:** each JSON lists `train_files` / `test_files` or IDs; enforce **no speaker in both**.
- **Utterance keys:** stable `utterance_id` matching waveform filename stem and one row in prosody manifest.
- **Reproducibility:** every run saves `seed`, library versions (optional `pip freeze`), and resolved config under `experiment/runs/<run_id>/`.

---

## 6. Current repo layout

Code follows this document: **AASIST-L** lives in `src/models/`, audio utilities in `src/features/`, attention demos in `experiment/scripts/`, and pytest suites under `experiment/tests/`. Use `pip install -e .` or `PYTHONPATH=src` for imports (`from models import Model`, `from features.pitch_extraction import …`).
