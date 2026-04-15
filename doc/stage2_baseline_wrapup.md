# Stage 2 Baseline Run — Wrap-Up Report
**Date:** 2026-04-15  
**Model:** AASIST-L + Prosody Late Fusion (`AASISTWithProsody`)  
**Dataset:** CSS Pilot (~108 samples, 5-fold speaker-disjoint CV)

---

## 1. What Was Done

This run completes **Stage 2** of the SpoofingDoo project: Transfer Learning of AASIST-L (pre-trained on ASVspoof 2019 English data) adapted for Thai speech spoofing detection via prosodic late fusion.

**Key steps executed:**
1. Loaded ASVspoof-pretrained `AASIST-L.pth` checkpoint (0 missing / 0 unexpected keys after architecture fix)
2. Froze the entire SincNet + Graph backbone (`requires_grad=False`) — only 4 tensors trainable
3. Trained the prosody MLP + readout head for up to 50 epochs with early stopping (patience=5)
4. Generated 4 visualizations per fold: training curves, score distribution, DET+ROC curve, style confusion matrix

---

## 2. Architecture Fix Applied

The `AASIST_L_ARGS` in `baseline_train.py` was corrected to match the actual checkpoint. The original code used `gat_dims=[64,32]` and `filts[3]=[32,64]`; the true AASIST-L checkpoint uses smaller filters:

| Parameter   | Wrong (original) | Correct (from .pth)  |
|-------------|-----------------|----------------------|
| `filts[3]`  | `[32, 64]`      | `[32, 24]`           |
| `filts[4]`  | `[64, 64]`      | `[24, 24]`           |
| `gat_dims`  | `[64, 32]`      | `[24, 32]`           |
| `mgo_dim`   | 160             | 160 (unchanged: 5×32)|

After the fix: **Missing: 0, Unexpected: 0** on all 5 folds.

**Trainable parameters (4 tensors, ~2.9k params):**
```
prosody_mlp.0.weight : [16, 3]
prosody_mlp.0.bias   : [16]
out_layer.weight     : [1, 176]   ← mgo_dim(160) + prosody_feat(16)
out_layer.bias       : [1]
```

---

## 3. Training Configuration

| Hyperparameter  | Value  |
|-----------------|--------|
| Optimizer       | Adam   |
| Learning Rate   | 1e-3   |
| LR Scheduler    | ReduceLROnPlateau (factor=0.5, patience=2) |
| Batch Size      | 16     |
| Max Epochs      | 50     |
| Early Stop Patience | 5  |
| Augmentation    | Gaussian noise (SNR=15 dB) |
| Loss            | BCEWithLogitsLoss (pos_weight=0.2 for 5:1 spoof:genuine) |

---

## 4. Cross-Validation Results

| Fold | Train / Val | Best EER (%) | AUC    | Best Epoch / Total |
|------|------------|--------------|--------|--------------------|
| 00   | 72 / 36    | 50.00        | 0.3889 | 1 / 6              |
| 01   | 90 / 18    | 33.33        | 0.6000 | 2 / 7              |
| 02   | 90 / 18    | 66.67        | 0.3778 | 3 / 8              |
| 03   | 90 / 18    | 33.33        | 0.5556 | 5 / 10             |
| 04   | 90 / 18    | 33.33        | 0.6222 | 2 / 7              |
| **Mean** | —    | **43.33**    | **0.5089** | —            |
| **±Std** | —    | **±13.33**   | **±0.1048** | —           |

---

## 5. Analysis & Interpretation

### 5.1 EER Granularity Is Dominated by Small Test Set
With only 18–36 test samples and a 5:1 spoof:genuine imbalance, the smallest possible non-trivial EER step is `1/n_genuine ≈ 33%`. This means the results cluster at {33%, 50%, 67%} — the metric resolution is too coarse to meaningfully rank configurations. **Conclusion: EER on the 100-sample pilot is uninformative beyond "is the model better than chance".**

### 5.2 Model Barely Learns Above Chance
- Mean EER = 43.33% (near chance = 50%)
- Mean AUC = 0.509 (near chance = 0.5)
- Early stopping fires after 6–10 epochs consistently — the 5-epoch patience is appropriate

**Root cause:** Only 4 tiny trainable layers (< 3k params) receive gradients. The frozen AASIST-L backbone extracts English-acoustic features that have limited transferability to Thai prosodic patterns at the ~100-sample scale.

### 5.3 Folds 01, 03, 04 Show Marginal Signal
These three folds achieved EER=33.33% and AUC between 0.56–0.62, indicating the model occasionally finds a weak decision boundary. This is consistent with the prosody features (jitter/shimmer/F0) capturing some genuine vs. spoof contrast in certain speaker splits.

### 5.4 Class Imbalance
`pos_weight=0.2` down-weights the spoof class. With 75 spoofed vs 15 genuine training samples (folds 01–04), the model's default behavior is to predict low spoof probability for everything — explaining the 50%+ EER in many folds.

---

## 6. Outputs Per Fold

Each `experiment/runs/stage2_fold{00..04}/` contains:
```
best_model.pt              ← best checkpoint (by val EER)
history.json               ← per-epoch metrics
figures/
  training_loss.png        ← train/val loss + AUC/EER curves
  score_distribution.png   ← KDE of genuine vs. spoof scores
  det_curve.png            ← DET + ROC curves with EER point
  style_confusion_matrix.png ← per-style accuracy breakdown
```

---

## 7. Known Bugs Fixed in This Run

| Issue | Fix |
|-------|-----|
| `AASIST_L_ARGS` mismatch → RuntimeError on `load_state_dict` | Corrected `filts` and `gat_dims` to match checkpoint tensor shapes |
| `seaborn.kdeplot` crashes on pandas ≥ 2.0 (`mode.use_inf_as_null` removed) | Replaced with `scipy.stats.gaussian_kde` + `matplotlib` directly |

---

## 8. Recommendations for Next Steps

### Immediate (Pilot Phase)
1. **Increase patience / epochs** — Try patience=10, epochs=100. The model might improve more slowly given the tiny gradient signal.
2. **Lower `pos_weight` toward 1.0** — The current 0.2 weighting suppresses the spoof signal entirely. Try `pos_weight=1.0` to force the model to learn both classes.
3. **Tune LR** — Try `lr=1e-4` to prevent the ReduceLROnPlateau scheduler from halving LR too aggressively in the first few epochs.
4. **Validate prosody normalization** — Check whether jitter/shimmer/F0 values are z-scored before being fed to the MLP. Large-scale prosody features may dominate early gradient steps.

### Phase 2 (if 10% CSS dataset is acquired)
See `doc/updated_css_roadmap_10pct.md` and the PHASE 2 section of `CLAUDE.md` — unfreeze the full backbone, discriminative LRs, train/val/test split instead of 5-fold CV.

---

## 9. Reproducibility

```bash
cd SpoofingDoo_Project
for fold in 00 01 02 03 04; do
    python3 experiment/scripts/baseline_train.py \
        --fold experiment/protocols/folds/fold_${fold}.json \
        --manifest data/features/prosody_manifest.json \
        --checkpoint data/pretrained/AASIST-L.pth \
        --out_dir experiment/runs/stage2_fold${fold} \
        --fig_dir experiment/runs/stage2_fold${fold}/figures \
        --epochs 50 --lr 1e-3 --batch_size 16 --patience 5
done
```
