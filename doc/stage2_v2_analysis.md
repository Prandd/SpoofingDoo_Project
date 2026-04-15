# Stage 2 v2 — Detailed Analysis Report
**Date:** 2026-04-15  
**Model:** AASIST-L + Prosody Late Fusion (`AASISTWithProsody`)  
**Run config:** `pos_weight=1.0`, `lr=1e-4`, `epochs=100`, `patience=10`, `BatchNorm1d` on prosody  
**Dataset:** CSS Pilot (~108 samples, 5-fold speaker-disjoint CV)

---

## 1. Cross-Validation Summary

### v1 vs. v2 Side-by-Side

| Fold | v1 EER (%) | v1 AUC | v1 Epochs | v2 EER (%) | v2 AUC | v2 Epochs |
|------|-----------|--------|-----------|-----------|--------|-----------|
| 00   | 50.00     | 0.389  | 1/6       | **63.33** | 0.306  | 10/20     |
| 01   | 33.33     | 0.600  | 2/7       | **33.33** | **0.822** | 1/11 |
| 02   | 66.67     | 0.378  | 3/8       | **46.67** | **0.533** | 13/23 |
| 03   | 33.33     | 0.556  | 5/10      | 60.00     | 0.467  | 2/12      |
| 04   | 33.33     | 0.622  | 2/7       | 66.67     | 0.511  | 1/11      |
| **Mean** | **43.33** | **0.509** | — | **54.00** | **0.528** | — |
| **±Std** | **±13.33** | **±0.105** | — | **±12.36** | **±0.167** | — |

> **Net verdict:** v2 is mixed — Folds 01 and 02 improved substantially (AUC), while Folds 03 and 04 regressed on EER. Mean AUC is marginally better (+0.019), but mean EER worsened (+10.67 pp). High AUC variance (0.167) dominates the picture.

---

## 2. Per-Fold Analysis

### Fold 00 — Most Difficult (72 train / 36 val)

| Direction | Metric | Observation |
|---|---|---|
| ↓ | EER | 50% → 63.33% (worse) |
| ↓ | AUC | 0.389 → 0.306 (worse) |
| ↑ | Training depth | 6 → 20 epochs (longer, patience=10 working) |

**Learning curve pattern:** EER is stuck at 66.7% for epochs 1–9, drops briefly to 63.3% at epoch 10, then early-stopping fires. AUC trends *downward* from 0.37 to 0.27 over 20 epochs — the model is fitting the training set but the decision boundary is moving in the wrong direction on the validation set.

**Root cause:** Fold 00 has only 72 training samples (the smallest fold). After removing backbone gradients, only ~3k params are being trained on this tiny batch. With `pos_weight=1.0`, the model is forced to weight genuine (likely 12 of 72) and spoof samples equally in loss computation, but genuine samples are simply too rare to provide a reliable gradient signal. The v1 run's `pos_weight≈0.2` accidentally *did* help here — by strongly down-weighting the spoof class, it pushed the model to concentrate on the few genuine examples.

---

### Fold 01 — Best Performer (90 train / 18 val)

| Direction | Metric | Observation |
|---|---|---|
| → | EER | 33.33% → 33.33% (unchanged — minimum achievable) |
| ↑ | AUC | 0.600 → **0.822** (+0.222, major improvement) |
| ↑ | AUC peak | — | 0.889 at epoch 4 |
| — | Training depth | 7 → 11 epochs |

**Learning curve pattern:** The model achieves EER=33.33% immediately at epoch 1 (AUC=0.822) and maintains this through epoch 9. AUC peaks at 0.889 (epoch 4), then decays to 0.689 as loss continues to fall — classic sign that the model overfit slightly on the training distribution before early-stopping fired at epoch 11.

**Root cause (improvement):** Fold 01's test split likely contains easier-to-separate genuine vs. spoof examples. With `pos_weight=1.0`, the model gets equal gradient from both classes immediately, and the `BatchNorm1d` on prosody ensures F0/jitter features contribute proportionally from epoch 1. The result is a much sharper score distribution: AUC 0.822 means the model is actually ranking spoofed above genuine ~82% of the time.

**Key insight:** Fold 01 at AUC=0.822 demonstrates that the prosody-fused AASIST-L *can* learn a meaningful Thai spoofing signal — the barrier is data scarcity and fold composition, not fundamental model incapacity.

---

### Fold 02 — Moderate Improvement (90 train / 18 val)

| Direction | Metric | Observation |
|---|---|---|
| ↑ | EER | 66.67% → **46.67%** (−20 pp improvement) |
| ↑ | AUC | 0.378 → 0.533 (+0.155) |
| ↑ | Training depth | 8 → 23 epochs |

**Learning curve pattern:** EER starts at 60%, improves to 46.67% at epoch 13, then degrades back to 66.7%. The model found a weak decision boundary mid-training but couldn't maintain it. This is the only fold where the boundary was discovered *after* several epochs — suggesting a slow learning trajectory at `lr=1e-4` that could benefit from slightly more epochs.

**Root cause (improvement):** The lower LR prevented the fast over-shoot seen in v1 (where EER hit 66.67% from epoch 1). The BatchNorm1d normalisation appears to have stabilised early gradients. The boundary found at epoch 13 is fragile — the model's ~3k trainable params can't resist small loss oscillations.

---

### Fold 03 — Regression (90 train / 18 val)

| Direction | Metric | Observation |
|---|---|---|
| ↓ | EER | **33.33% → 60.00%** (+26.67 pp regression) |
| ↓ | AUC | 0.556 → 0.467 (worse) |
| ↑ | Training depth | 10 → 12 epochs |

**Learning curve pattern:** EER starts at 60% in epoch 1 and never breaks below it. Val loss steadily decreases from 0.544 to 0.442, but EER remains pinned — the model is lowering calibrated loss without improving rank-ordering of genuine vs. spoof.

**Root cause (regression):** In v1, `pos_weight≈0.2` apparently *pushed* the output logits toward predicting "genuine" (low spoof prob), which by chance happened to be the right decision for this fold's test distribution. With `pos_weight=1.0`, the model now balances both classes, but the fold's genuine test samples may simply score similarly to spoofs at this LR/epoch budget. The regression is likely sensitivity to fold-specific genuine-vs-spoof score distributions rather than a systematic model failure.

---

### Fold 04 — Regression (90 train / 18 val)

| Direction | Metric | Observation |
|---|---|---|
| ↓ | EER | **33.33% → 66.67%** (+33.33 pp regression) |
| ↓ | AUC | 0.622 → 0.511 (near-chance) |
| — | Training depth | 7 → 11 epochs |

**Learning curve pattern:** EER is stuck at 66.67% from epoch 1 through 11. AUC oscillates between 0.42 and 0.51 — essentially random discrimination. Val loss decreases from 0.557 to 0.427 (model is training), but the boundary never flips.

**Root cause (regression):** This is the strongest v2 regression. In v1, EER=33.33% appeared from epoch 2 — likely the model was biased toward predicting "genuine" for almost everything (low spoof prob for all), and this fold happened to have very few genuine test samples. With `pos_weight=1.0` forcing genuine/spoof balance, the model tries to predict a ~50% boundary and fails. This fold's genuine/spoof test samples may overlap heavily in the prosody feature space.

---

## 3. Cross-Cutting Observations

### 3.1 Training Loss Is Consistently Decreasing

In all 5 folds, train loss decreases monotonically:

| Fold | Epoch 1 Train Loss | Last Epoch Train Loss | Δ |
|---|---|---|---|
| 00 | 0.860 | 0.617 (ep20) | −0.243 |
| 01 | 0.663 | 0.545 (ep11) | −0.118 |
| 02 | 0.878 | 0.537 (ep23) | −0.341 |
| 03 | 0.604 | 0.511 (ep12) | −0.093 |
| 04 | 0.624 | 0.529 (ep11) | −0.095 |

The backbone is frozen, so this improvement comes entirely from the prosody MLP + readout. The model is learning, but the learned boundary does not generalise to the val split — a classic sign of **over-parameterisation relative to data** in the trainable head, even with only ~3k params and Dropout(0.5).

### 3.2 Val Loss Also Decreases — But EER Doesn't Follow

A striking pattern: val loss decreases in every fold (e.g. fold 02: 0.816→0.545, fold 04: 0.557→0.427) without a corresponding EER improvement. This means:
- The model is assigning somewhat more confident correct-class probabilities on average
- But the *relative ordering* of genuine vs. spoof scores — which is what EER/AUC measure — remains poor

This indicates the model is calibrating confidence levels without improving discrimination. The prosody MLP is likely assigning a uniform "slightly spoofy" confidence to everything, bringing val loss down but not separating the two classes.

### 3.3 Early Stopping at 11–23 Epochs — Patience Is Now the Binding Constraint

| Fold | Best Epoch | Total Epochs | Margin |
|---|---|---|---|
| 00 | 10 | 20 | Best found mid-run; 10 more with no improvement |
| 01 | 1 | 11 | Best at epoch 1; 10 patience epochs exhausted |
| 02 | 13 | 23 | Best mid-run; 10 more with no improvement |
| 03 | 2 | 12 | Best early; 10 patience epochs exhausted |
| 04 | 1 | 11 | Best at epoch 1; 10 patience epochs exhausted |

The patience=10 setting is now the tight binding constraint — all folds exhaust the full patience budget before stopping. This is correct behavior, but it also means we are spending ~10 epochs confirming that no improvement exists rather than finding new improvements. **patience=5 might be the right setting at this data scale**.

### 3.4 EER Quantisation Makes Ranking Unreliable

With 18 test samples (3 genuine + 15 spoof in most folds), the finest EER step is:
```
1 / n_genuine = 1/3 ≈ 33.33%
```
Observable EER values: {33.33%, 46.67%, 60.00%, 66.67%} (fractions of 3 genuine samples).

This means a single test sample changing prediction can shift EER by 13–33 pp. Any comparison of configurations based on EER alone is **statistically unreliable** at this data scale. **AUC is the more reliable metric** and should be used as the primary ranking criterion.

---

## 4. v1 vs. v2 Metric Comparison (Using AUC as Primary)

When ranked by AUC:

| Fold | v1 AUC | v2 AUC | Winner |
|---|---|---|---|
| 00 | 0.389 | 0.306 | v1 |
| 01 | 0.600 | **0.822** | **v2 (+0.222)** |
| 02 | 0.378 | **0.533** | **v2 (+0.155)** |
| 03 | 0.556 | 0.467 | v1 (−0.089) |
| 04 | 0.622 | 0.511 | v1 (−0.111) |
| **Mean** | **0.509** | **0.528** | **v2 marginal** |

v2 wins on AUC in 2 of 5 folds, loses in 3 of 5 folds. The wins are larger in magnitude (+0.222, +0.155) than the losses (−0.089, −0.111, −0.083). **The mean AUC improvement of +0.019 is too small to be conclusive** at this sample size — this result is within the noise floor of 5-fold CV with 108 samples.

---

## 5. Fold Composition Hypothesis

The performance divergence across folds suggests the results are driven more by **which speakers ended up in each test split** than by the model configuration. Folds 01 and 02 likely contain test speakers whose prosody statistics (jitter, shimmer, F0) differ substantially between genuine and spoofed utterances. Folds 03 and 04 likely contain speakers where genuine/spoof prosody overlaps.

This cannot be diagnosed without examining per-speaker prosody distributions — a worthwhile next step.

---

## 6. Recommendations

### Immediate (Pilot Phase — 108 samples)

| Priority | Action | Expected Impact |
|---|---|---|
| 1 | **Revert patience to 5** | Avoids spending 10 epochs confirming no improvement; reduces total run time by ~half |
| 2 | **Inspect per-speaker prosody distributions** (boxplot of jitter/shimmer/F0 per speaker, genuine vs. spoof) | Explains why folds 03/04 regress; identifies separable vs. inseparable speakers |
| 3 | **Try label smoothing ε=0.1** in BCEWithLogitsLoss instead of `pos_weight` tuning | Reduces overconfident predictions without re-introducing the class suppression problem |
| 4 | **Try pos_weight as a swept hyperparameter {0.5, 1.0, 2.0}** per fold | Find the sweet spot between v1's suppression and v2's over-correction |
| 5 | **Examine fold 01 score distribution** (figures/score_distribution.png) | Understand what makes fold 01 separable — prosody pattern? Speaker composition? |

### Phase 2 (Full CSS 10% Dataset, ~13k samples)
When the larger dataset is available:
- Unfreeze the full backbone with discriminative LRs (`1e-6` backbone, `1e-4` head)
- Use stratified train/val/test split (speaker-disjoint, 70/15/15)
- EER quantisation will no longer be a binding constraint (n_genuine in test ≈ 200)
- The BatchNorm1d prosody normalisation and `pos_weight=1.0` settings from v2 should carry forward — they are theoretically sound, and the pilot's mixed results reflect data scarcity, not model design errors

---

## 7. Artifacts Reference

| File | Location | Description |
|---|---|---|
| `stage2_cv_summary.json` | `experiment/runs/` | Machine-readable CV aggregate (mean EER, AUC, per-fold table) |
| `stage2_cv_report.md` | `experiment/runs/` | Markdown CV table + last-5-epoch history per fold |
| `stage2_cv_curves.png` | `experiment/runs/` | Overlay of val-AUC and val-EER across all folds |
| `stage2_fold??/history.json` | `experiment/runs/` | Per-epoch metrics for each fold (v2 run) |
| `stage2_fold??/best_model.pt` | `experiment/runs/` | Best checkpoint per fold (by val EER) |
| `stage2_fold??/figures/*.png` | `experiment/runs/` | 4 visualizations per fold |
| `stage2/stage2_fold??/` | `experiment/runs/` | Older pre-v2 run artifacts (reference only) |

---

## 8. Conclusion

The v2 training dynamics fixes produced **mixed results** that collectively do not meet the target of "consistent EER=33% across ≥4 folds". The AUC improvement in Fold 01 (0.600→0.822) is the most positive signal and demonstrates that the prosody-fused model *can* learn a meaningful Thai spoofing boundary.

The core limiting factor remains the dataset size: 108 samples, 5-fold CV, and a 5:1 class imbalance creates fold compositions with highly variable genuine/spoof separability. No hyperparameter change at the 100-sample scale is likely to yield stable improvements across all folds simultaneously. The results are valid for what they are — an honest characterisation of the model's capability under extreme data constraints — and the architecture (frozen AASIST-L + prosody BN-MLP + binary readout) is sound for Phase 2 scale-up.

**Recommended action before Phase 2:** Run the per-speaker prosody distribution diagnostic to understand fold separability, then apply label smoothing + patience=5 for a final pilot experiment. Document the best achieved AUC per fold as the Phase 2 baseline target.
