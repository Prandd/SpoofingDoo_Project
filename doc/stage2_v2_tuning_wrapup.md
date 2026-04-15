# Stage 2 v2 — Training Dynamics Fix & Tooling Wrap-Up
**Date:** 2026-04-15  
**Model:** AASIST-L + Prosody Late Fusion (`AASISTWithProsody`)  
**Dataset:** CSS Pilot (~108 samples, 5-fold speaker-disjoint CV)  
**Status:** Code changes applied; re-run pending.

---

## 1. Changes Applied vs. Stage 2 v1

This document covers the three targeted fixes applied after the Stage 2 baseline post-mortem
(see `doc/stage2_baseline_wrapup.md` for the original baseline run details).

### 1.1 Fix: `pos_weight = 1.0` (was dynamic ~0.2)

**File:** `experiment/scripts/baseline_train.py` — `compute_pos_weight()`

| | v1 (broken) | v2 (fixed) |
|---|---|---|
| `pos_weight` | `n_genuine / n_spoof ≈ 0.2` | `1.0` (hardcoded) |
| Effect | BCE down-weights every spoof gradient by 5× → model predicts low spoof probability for everything → EER ≈ 50% | Equal loss weight for both classes; model is forced to learn a decision boundary |

Root cause: With 75 spoof vs 15 genuine samples, dynamic `pos_weight` was intended to up-weight the minority (genuine) class — but the convention in `BCEWithLogitsLoss` is that `pos_weight` scales the **positive (spoof=1)** class, making the 0.2 value actively harmful at this scale.

### 1.2 Fix: Longer Training Budget — `epochs=100`, `patience=10`

**File:** `experiment/scripts/baseline_train.py` — `parse_args()`

| Hyperparameter | v1 | v2 |
|---|---|---|
| `--epochs` default | 50 | 100 |
| `--patience` default | 5 | 10 |

The v1 model consistently stopped at 6–10 epochs. With only ~3k trainable parameters
receiving a tiny gradient signal, the model may need many more iterations to
find a weak but non-trivial decision boundary.

### 1.3 Fix: Lower Learning Rate — `lr=1e-4` (was `1e-3`)

**File:** `experiment/scripts/baseline_train.py` — `parse_args()`

`ReduceLROnPlateau(factor=0.5, patience=2)` with an initial LR of `1e-3` halved
the LR to `2.5e-4` within the first 3 epochs, often before the model had a chance
to orient itself. Starting at `1e-4` keeps the effective LR in a stable range
for longer.

### 1.4 Fix: `BatchNorm1d` before prosody Linear layer

**File:** `src/models/aasist_imported.py` — `AASISTWithProsody.__init__()`

```python
# v1
self.prosody_mlp = nn.Sequential(
    nn.Linear(prosody_dim, 16),
    nn.ReLU(),
    nn.Dropout(p=0.5),
)

# v2
self.prosody_mlp = nn.Sequential(
    nn.BatchNorm1d(prosody_dim),   # ← NEW: auto Z-score normalisation
    nn.Linear(prosody_dim, 16),
    nn.ReLU(),
    nn.Dropout(p=0.5),
)
```

Jitter values (~0.01–0.05) and F0 standard deviation (~20–80 Hz) differ by
~3 orders of magnitude. Without normalisation, early gradients are dominated
by the largest-scale feature, preventing the MLP from learning balanced
representations. `BatchNorm1d` handles this automatically at the cost of
two extra trainable scalars (γ, β) per prosody dimension.

**Trainable parameter count (v2):**
```
prosody_mlp.0.weight  : [3]    ← BN gamma
prosody_mlp.0.bias    : [3]    ← BN beta
prosody_mlp.1.weight  : [16, 3]
prosody_mlp.1.bias    : [16]
out_layer.weight      : [1, 176]
out_layer.bias        : [1]
```
Total: 6 tensors, ~3k params (was 4 tensors).

---

## 2. New Scripts

### 2.1 `experiment/scripts/run_folds.sh` — 5-fold CV runner

Runs all five folds sequentially, writes per-fold logs to
`experiment/runs/stage2_cv_logs/fold_??.log`, and emits a summary CSV.

```bash
# From SpoofingDoo_Project root:
bash experiment/scripts/run_folds.sh

# With custom checkpoint / prefix:
CHECKPOINT=data/pretrained/AASIST-L.pth \
OUT_PREFIX=experiment/runs/stage2_v2 \
bash experiment/scripts/run_folds.sh
```

Key env-var overrides:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT` | `data/pretrained/AASIST-L.pth` | Backbone weights |
| `MANIFEST` | `data/features/prosody_manifest.json` | Prosody feature index |
| `OUT_PREFIX` | `experiment/runs/stage2` | Fold output dir prefix |
| `EPOCHS` | `100` | Max epochs per fold |
| `LR` | `1e-4` | Learning rate |
| `PATIENCE` | `10` | Early-stop patience |
| `BATCH` | `16` | Batch size |

### 2.2 `experiment/scripts/collect_results.py` — Result aggregator

Reads every `<prefix>_fold??/history.json` and produces three outputs:

| Output | Description |
|---|---|
| `<prefix>_cv_summary.json` | Machine-readable mean/std EER & AUC + per-fold table |
| `<prefix>_cv_report.md` | Human-readable Markdown table with last-5-epoch training history |
| `<prefix>_cv_curves.png` | Overlay plot of val-AUC and val-EER across all folds |

```bash
python3 experiment/scripts/collect_results.py \
    --runs_root experiment/runs \
    --prefix    stage2
```

---

## 3. Visualization Requirements (Confirmed Met)

`baseline_train.py` generates all 4 required figures per fold in `<out_dir>/figures/`:

| File | Content | Status |
|---|---|---|
| `training_loss.png` | Train vs. Val Loss (left) + Val AUC & EER dual-axis (right) | ✓ |
| `score_distribution.png` | KDE of genuine vs. spoof score distributions with EER threshold marker | ✓ |
| `det_curve.png` | DET curve with EER point (left) + ROC curve with AUC (right) | ✓ |
| `style_confusion_matrix.png` | Per-style confusion counts bar chart (left) + accuracy heatmap (right) | ✓ |

Known fix already in place: `scipy.stats.gaussian_kde` is used directly
(instead of `seaborn.kdeplot`) to avoid the `mode.use_inf_as_null` deprecation
on pandas ≥ 2.0.

---

## 4. How to Re-Run

```bash
cd SpoofingDoo_Project

# Option A: shell loop (recommended — uses all env overrides)
bash experiment/scripts/run_folds.sh

# Option B: manual per-fold (useful for debugging a single fold)
python3 experiment/scripts/baseline_train.py \
    --fold       experiment/protocols/folds/fold_00.json \
    --manifest   data/features/prosody_manifest.json \
    --checkpoint data/pretrained/AASIST-L.pth \
    --out_dir    experiment/runs/stage2_fold00 \
    --fig_dir    experiment/runs/stage2_fold00/figures \
    --epochs 100 --lr 1e-4 --batch_size 16 --patience 10

# Collect results after all folds finish:
python3 experiment/scripts/collect_results.py \
    --runs_root experiment/runs \
    --prefix    stage2
```

---

## 5. Expected Impact

| Change | Expected Effect |
|---|---|
| `pos_weight=1.0` | Biggest lever — eliminates the class suppression that caused EER≈50% in 2 of 5 folds |
| Lower LR (`1e-4`) | More stable convergence before ReduceLROnPlateau kicks in |
| Longer budget (100 ep / patience 10) | Allows gradual boundary formation with ~3k trainable params |
| `BatchNorm1d` | Prevents F0 scale dominance in early gradient steps; should reduce variance across folds |

Given the coarse EER resolution (33%/50%/67% steps with ~18 test samples),
the primary success criterion is **consistent EER=33% across ≥ 4 folds** with
**mean AUC > 0.60**.

---

## 6. Known Limitations (Unchanged from v1)

- With only 18–36 test samples per fold, EER resolution is ~33% increments.
  Continuous metrics (AUC) are more informative than EER at this scale.
- The frozen backbone extracts English-acoustic features. Full Thai domain
  adaptation requires the Phase 2 dataset (~133k samples).
- 5-fold speaker-disjoint CV with 108 samples yields highly variable folds
  (fold 00 has only 72 train samples vs. 90 for folds 01–04).
