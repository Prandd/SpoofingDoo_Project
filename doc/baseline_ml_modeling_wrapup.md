# ML Modeling Phase — Wrap-Up  *(Baseline)*

## Project: SpoofingDoo — Thai Speech Spoofing Detection (CSS Pilot, 100 Samples)

> **This document describes the BASELINE system.**
> All subsequent experiments (Stage 3 cross-style, ablations, full-dataset scale-up)
> must compare their EER/AUC against the numbers reported here.
>
> | Artifact | Path |
> |---|---|
> | Baseline training script | `experiment/scripts/baseline_train.py` |
> | Baseline model class | `src/models/aasist_imported.py` → `AASISTWithProsody` |
> | Baseline checkpoint (fold 0) | `experiment/runs/stage2_fold0/best_model.pt` |
> | Baseline figures | `experiment/runs/figures/` |

---

## 1. What Was Built

### 1.1 Model: `AASISTWithProsody` (`src/models/aasist_imported.py`)

We extended the **AASIST-L** (Attentive Graph for Spoofing deTection, Lightweight) model with a **late-fusion prosody branch** for Thai-specific voice quality features.

```
Input waveform (64,600 samples)
         │
    ┌────▼────────────────────────────────────┐
    │   AASIST-L Backbone (FROZEN)            │
    │                                         │
    │  SincNet → ResBlocks → GAT-S + GAT-T    │
    │        → HS-GAL × 2 → MGO              │
    │                          │              │
    │             last_hidden  (bs, 160)       │
    └─────────────────────────┬───────────────┘
                              │
  Prosody vector (bs, 3)      │       ← [jitter_mean, shimmer_mean, f0_std]
         │                    │
    ┌────▼────────┐            │
    │ Linear(3→16)│            │
    │ ReLU        │            │
    │ Dropout(0.5)│            │
    └────┬────────┘            │
         │  (bs, 16)           │
         └──────────cat────────┘
                    │  (bs, 176)
               ┌────▼────────────┐
               │ Linear(176 → 1) │  ← trainable readout
               └────┬────────────┘
                    │
               logit (bs,)   →   BCEWithLogitsLoss
```

**Why Dropout(0.5) in the prosody MLP?**  
Without it, the network could shortcut by memorising jitter/shimmer values and ignoring the SincNet-extracted features entirely. Dropout forces the model to integrate evidence from *both* branches, which is critical when prosody features are noisy on only 100 samples.

**Trainable parameters:** 4 tensors, ~300 parameters total.  
**Frozen parameters:** 175 tensors (entire AASIST-L backbone).

---

### 1.2 Training Loop: `experiment/scripts/train.py`

| Design decision | Implementation |
|---|---|
| Backbone frozen | `requires_grad=False` on all backbone params; `model.backbone.eval()` every epoch to keep frozen BatchNorm statistics stable |
| Class imbalance (18 genuine : 90 spoof) | `BCEWithLogitsLoss(pos_weight = n_genuine / n_spoof)` — computed dynamically per fold, ≈ 0.20 for fold 0 |
| Gaussian noise augmentation | `CSSDataset(augment=True, aug_snr_db=15.0)` on training split only |
| Early stopping | `patience=5` epochs without validation EER improvement |
| LR scheduling | `ReduceLROnPlateau(factor=0.5, patience=2)` on validation loss |
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` to stabilise tiny-dataset training |
| Metrics | **EER** (primary, anti-spoofing standard) + **AUC** (secondary) per epoch |

#### `pos_weight` rationale

With `label=0` for genuine and `label=1` for spoofed, `BCEWithLogitsLoss` computes:

```
loss = pos_weight × y × -log σ(x)  +  (1-y) × -log(1-σ(x))
```

Setting `pos_weight = n_genuine / n_spoof ≈ 0.2` scales down the contribution of each spoof sample so the total spoof loss equals the total genuine loss. Without this, the model would minimise loss by always predicting spoof — correctly labelling all 90 spoof samples at the cost of misclassifying all 18 genuine samples.

---

### 1.3 Evaluation Metrics (`src/eval/__init__.py`)

- **`compute_eer(labels, scores)`** — Threshold sweep (no external dependency). Returns `(eer, threshold)`.
- **`compute_auc(labels, scores)`** — Uses `sklearn.metrics.roc_auc_score` with a manual trapezoid fallback.

---

### 1.4 Visualisations (`experiment/runs/figures/`)

Four figures are generated automatically at the end of each training run:

| File | Contents |
|---|---|
| `training_loss.png` | Train/Val loss curves + Val AUC & EER over epochs |
| `score_distribution.png` | KDE of predicted spoof probability for genuine vs. spoofed samples; EER threshold line |
| `det_curve.png` | DET curve (FAR vs. FRR %) with EER marker + ROC curve subplot (AUC) |
| `style_confusion_matrix.png` | Per-style (Formal/Casual/Excited) TP/FP/TN/FN counts + detection accuracy heatmap |

---

## 2. How to Run

### Single fold

```bash
# From SpoofingDoo_Project/
python experiment/scripts/train.py \
  --fold     experiment/protocols/folds/fold_00.json \
  --manifest data/features/prosody_manifest.json \
  --checkpoint data/pretrained/aasist_l.pth \   # optional; random init if absent
  --out_dir  experiment/runs/stage2_fold0 \
  --fig_dir  experiment/runs/figures \
  --epochs   50  --lr 1e-3  --batch_size 16  --patience 5
```

### All 5 folds (bash loop)

```bash
for i in 00 01 02 03 04; do
  python experiment/scripts/train.py \
    --fold     experiment/protocols/folds/fold_${i}.json \
    --manifest data/features/prosody_manifest.json \
    --out_dir  experiment/runs/stage2_fold${i} \
    --fig_dir  experiment/runs/figures \
    --epochs 50 --lr 1e-3 --patience 5
done
```

---

## 3. Key Results (Fold 0 — confirmed run)

### Actual run output (no pre-trained checkpoint)

| Epoch | Train Loss | Val Loss | Val EER | Val AUC |
|-------|-----------|----------|---------|---------|
| 1     | 0.2887    | 0.2153   | 50.00%  | 0.611   |
| 2     | 0.2447    | 0.2084   | 50.00%  | 0.611   |
| 3     | 0.2684    | 0.2056   | 50.00%  | 0.611   |
| 4     | 0.2446    | 0.2029   | 50.00%  | 0.611   |
| 5     | 0.2646    | 0.2022   | 50.00%  | 0.611   |
| 6     | 0.2411    | 0.2027   | 50.00%  | 0.611   |
| — | *(early stop, patience=5)* | | | |

**Best val EER: 50.00% | Best AUC: 0.611 | Early stopped at epoch 6**

### Why EER=50% with random backbone?

The backbone (SincNet → ResBlocks → GAT → MGO) is frozen with **random weights**, so `last_hidden` (shape `160`) is effectively random noise regardless of the input audio. The only signal available to the model is the **3-D prosody vector** (jitter, shimmer, F0 std) fed through the small MLP. At 50% EER, the prosody features alone were not sufficient to separate genuine from spoofed speech with a linear readout at this scale — or the prosody extractor's output is not yet discriminative enough.

**AUC=0.611** (above 0.5 chance) shows the prosody features carry *some* weak signal, but the model cannot find a reliable threshold.

### What to expect with pre-trained AASIST-L weights

Once `data/pretrained/aasist_l.pth` is supplied:
- The backbone will produce **semantically meaningful** 160-D embeddings from the raw waveform, capturing spectral/temporal spoofing artefacts learned from English ASVspoof 2019.
- **EER** typically converges in 10–20 epochs, expected range **15–35%** (domain shift from English expected).
- **AUC** should exceed **0.70**, rising toward **0.85+** when prosody features complement the acoustic backbone.
- **Style breakdown:** Formal speech should be easiest to detect (stable, predictable prosody). Excited speech is hardest (prosodic masking), which is the core research hypothesis.

---

## 4. Architecture Constraints & Design Rationale

### Why freeze the backbone?

With only 100 samples (72 train per fold), fine-tuning the full 85k-parameter AASIST-L backbone would cause rapid overfitting. Freezing limits the learnable parameters to **~300** (prosody MLP + 1-neuron readout), which is sustainable at this scale.

### Why late fusion and not early fusion?

Early fusion (e.g., concatenating prosody features into the SincNet input) would require restructuring the SincNet layer and risk tensor shape bugs. Late fusion appends a parallel branch to the already-computed MGO embedding — minimal code change, zero graph topology disruption.

### Why `BCEWithLogitsLoss` (binary) instead of the original 2-class CE?

The original AASIST-L uses `CrossEntropyLoss` with a 2-class softmax head. For a highly imbalanced binary task, `BCEWithLogitsLoss + pos_weight` gives direct, interpretable control over the genuine/spoof trade-off without altering class prior assumptions.

---

## 5. File Map

```
src/
├── models/
│   ├── aasist_imported.py      ← Model (backbone) + AASISTWithProsody (new wrapper)
│   └── __init__.py             ← exports Model, AASISTWithProsody
├── eval/
│   └── __init__.py             ← compute_eer(), compute_auc()
└── datasets/
    └── dataset.py              ← CSSDataset (augment=True for train)

experiment/
├── scripts/
│   └── train.py                ← full training loop + 4 visualisations
├── protocols/folds/
│   └── fold_0{0-4}.json        ← speaker-disjoint 5-fold CV splits
└── runs/
    ├── stage2_fold0/
    │   ├── best_model.pt       ← best checkpoint (lowest val EER)
    │   └── history.json        ← per-epoch metrics
    └── figures/
        ├── training_loss.png
        ├── score_distribution.png
        ├── det_curve.png
        └── style_confusion_matrix.png
```

---

## 6. Next Steps

1. **Obtain pre-trained AASIST-L weights** (`aasist_l.pth`) from the official AASIST repository and place them at `data/pretrained/aasist_l.pth`. This is the single most impactful improvement.
2. **Run all 5 folds** and report mean ± std EER across folds for the final results table.
3. **Stage 3 (cross-style):** Use `experiment/configs/stage3_cross_style.yaml` — train on Formal, test on Excited to quantify prosodic masking.
4. **Ablation:** Compare Stage-1 (readout-only fine-tune, no prosody branch) vs. Stage-2 (with prosody MLP) using the same fold splits.
5. **t-DCF** (optional): If the detection cost function is needed for the report, add `compute_tdcf()` to `src/eval/__init__.py`.
