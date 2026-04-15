"""BASELINE — Stage-2 Training: AASIST-L + Prosody Late Fusion (CSS pilot, 100 samples).

This script produces the **baseline** results for the SpoofingDoo project.
All future experiments (Stage 3 cross-style, ablations, full-dataset scale-up)
compare their EER/AUC against the checkpoint saved by this script.

Usage
-----
    # From SpoofingDoo_Project directory:
    python experiment/scripts/baseline_train.py \\
        --fold experiment/protocols/folds/fold_00.json \\
        --manifest data/features/prosody_manifest.json \\
        --checkpoint data/pretrained/aasist_l.pth \\
        --out_dir experiment/runs/baseline_fold0 \\
        --epochs 50 --lr 1e-3 --batch_size 16 --patience 5

CRITICAL design points
----------------------
- SincNet and all graph layers are COMPLETELY FROZEN (requires_grad=False).
- ONLY AASISTWithProsody.prosody_mlp and .out_layer are trainable.
- BCEWithLogitsLoss with pos_weight = 1.0 (fixed). Dynamic computation gave
  ~0.2 which suppressed spoof learning entirely at the 100-sample scale.
- Gaussian noise augmentation is applied at the dataset level (augment=True).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project imports ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from models import Model, AASISTWithProsody
from datasets.dataset import CSSDataset, load_manifest
from eval import compute_eer, compute_auc


# ── Default AASIST-L architecture args (lightweight ~85k-param variant) ────────
# Derived from the actual AASIST-L.pth checkpoint tensor shapes:
#   encoder filts[3]=[32,24], filts[4]=[24,24], gat_dims=[24,32]
#   out_layer: [2, 160] = 5 * 32 → mgo_dim=160
AASIST_L_ARGS: dict = {
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
    "first_conv": 128,
}


# ══════════════════════════════════════════════════════════════════════════════
# Model helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_model(checkpoint: Path | None, device: torch.device) -> AASISTWithProsody:
    """Load AASIST-L backbone (optionally from checkpoint) and wrap it."""
    backbone = Model(d_args=AASIST_L_ARGS)

    if checkpoint is not None and checkpoint.exists():
        state = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"[Checkpoint] Loaded {checkpoint.name}. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        msg = (f"Checkpoint not found at {checkpoint}." if checkpoint
               else "No checkpoint provided.")
        print(f"[Warning] {msg} Using random backbone weights.")

    model = AASISTWithProsody(backbone=backbone, prosody_dim=3).to(device)
    return model


def compute_pos_weight(fold: dict, device: torch.device) -> torch.Tensor:
    """Force pos_weight=1.0 to prevent the 0.2 weight from suppressing spoof learning.

    With ~75 spoof : 15 genuine (5:1 ratio), dynamic pos_weight would be 0.2,
    which causes the model to always predict low spoof probability → EER ≈ 50%.
    Fixing to 1.0 forces the model to learn both classes equally.
    """
    labels = list(fold["train_labels"].values())
    n_spoof   = sum(1 for l in labels if l == 1)
    n_genuine = sum(1 for l in labels if l == 0)
    print(f"[Imbalance] train genuine={n_genuine}, spoof={n_spoof} "
          f"→ pos_weight=1.0 (fixed; dynamic would be {n_genuine/max(n_spoof,1):.4f})")
    return torch.tensor([1.0], device=device)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model: AASISTWithProsody,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Returns (avg_loss, EER, AUC)."""
    model.eval()
    all_scores: List[float] = []
    all_labels: List[int]   = []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for waveforms, prosody, labels, _ in loader:
            waveforms = waveforms.to(device)
            prosody   = prosody.to(device)
            labels_f  = labels.float().to(device)

            logits = model(waveforms, prosody, Freq_aug=False)
            total_loss += criterion(logits, labels_f).item()
            n_batches  += 1

            all_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss   = total_loss / max(n_batches, 1)
    labels_arr = np.array(all_labels,  dtype=np.int32)
    scores_arr = np.array(all_scores,  dtype=np.float32)
    eer, _     = compute_eer(labels_arr, scores_arr)
    auc        = compute_auc(labels_arr, scores_arr)
    return avg_loss, eer, auc


def evaluate_with_metadata(
    model: AASISTWithProsody,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Full evaluation pass that also collects per-sample metadata (style, utt_id).

    Returns a dict with:
        scores   : np.ndarray (N,)  — spoof probability in [0,1]
        labels   : np.ndarray (N,)  — 0=genuine, 1=spoof
        styles   : list[str] (N,)   — Formal / Casual / Excited
        utt_ids  : list[str] (N,)
    """
    model.eval()
    scores_list:  List[float] = []
    labels_list:  List[int]   = []
    styles_list:  List[str]   = []
    utt_ids_list: List[str]   = []

    with torch.no_grad():
        for waveforms, prosody, labels, meta in loader:
            waveforms = waveforms.to(device)
            prosody   = prosody.to(device)
            logits    = model(waveforms, prosody, Freq_aug=False)
            probs     = torch.sigmoid(logits).cpu().numpy()

            scores_list.extend(probs.tolist())
            labels_list.extend(labels.numpy().tolist())
            styles_list.extend(meta["style"])
            utt_ids_list.extend(meta["utt_id"])

    return {
        "scores":  np.array(scores_list,  dtype=np.float32),
        "labels":  np.array(labels_list,  dtype=np.int32),
        "styles":  styles_list,
        "utt_ids": utt_ids_list,
    }


def compute_det_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute FAR and FRR arrays for a DET curve by sweeping thresholds.

    Sweep descending: start with threshold=max (nothing accepted as spoof),
    lower it step by step to include each next-highest-scored sample.
      - FAR (False Alarm Rate) = FP / n_genuine   (genuine wrongly flagged)
      - FRR (False Reject Rate) = FN / n_spoof     (spoof missed)
    EER is where FAR ≈ FRR.

    Returns (far, frr, eer_value).
    """
    sorted_idx    = np.argsort(scores)[::-1]   # descending
    sorted_labels = labels[sorted_idx]
    n_pos         = int(labels.sum())           # n_spoof
    n_neg         = len(labels) - n_pos         # n_genuine

    tp = fp = 0
    # Seed point: threshold above everything → nothing accepted → FAR=0, FRR=1
    far_list: List[float] = [0.0]
    frr_list: List[float] = [1.0]

    for lbl in sorted_labels:
        if lbl == 1:
            tp += 1          # spoof correctly flagged
        else:
            fp += 1          # genuine wrongly flagged
        far_list.append(fp / max(n_neg, 1))
        frr_list.append(1.0 - tp / max(n_pos, 1))

    far_arr = np.array(far_list, dtype=np.float32)
    frr_arr = np.array(frr_list, dtype=np.float32)

    diff    = np.abs(far_arr - frr_arr)
    eer_idx = int(np.argmin(diff))
    eer_val = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    return far_arr, frr_arr, eer_val


# ══════════════════════════════════════════════════════════════════════════════
# Visualization functions
# ══════════════════════════════════════════════════════════════════════════════

def _setup_matplotlib() -> None:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend, safe for all environments


def plot_training_curves(history: List[dict], fig_dir: Path) -> None:
    """1. training_loss.png — loss + AUC over epochs."""
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")

    epochs     = [r["epoch"]      for r in history]
    tr_loss    = [r["train_loss"] for r in history]
    val_loss   = [r["val_loss"]   for r in history]
    val_auc    = [r["val_auc"]    for r in history]
    val_eer    = [r["val_eer"] * 100 for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss subplot
    axes[0].plot(epochs, tr_loss,  marker="o", ms=4, label="Train Loss",  color="#4c72b0")
    axes[0].plot(epochs, val_loss, marker="s", ms=4, label="Val Loss",    color="#dd8452")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Training vs. Validation Loss")
    axes[0].legend()

    # AUC + EER subplot
    color_auc = "#4c72b0"
    color_eer = "#c44e52"
    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax1.plot(epochs, val_auc, marker="o", ms=4, label="Val AUC",  color=color_auc)
    ax2.plot(epochs, val_eer, marker="^", ms=4, label="Val EER (%)", color=color_eer,
             linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("AUC",    color=color_auc)
    ax2.set_ylabel("EER (%)", color=color_eer)
    ax1.set_title("Validation AUC & EER")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    out_path = fig_dir / "training_loss.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[Figure] {out_path}")


def plot_score_distribution(eval_result: dict, fig_dir: Path) -> None:
    """2. score_distribution.png — KDE of spoof-score by class.

    Uses scipy.stats.gaussian_kde directly to avoid seaborn/pandas version
    incompatibilities with mode.use_inf_as_null that affect older seaborn.
    """
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    scores  = eval_result["scores"]
    labels  = eval_result["labels"]

    genuine_scores = scores[labels == 0]
    spoof_scores   = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#f9f9f9")
    ax.grid(True, linestyle="--", alpha=0.5)

    x_range = np.linspace(0, 1, 300)

    def _plot_class(data, color, label):
        if len(data) > 1:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data, bw_method="scott")
            density = kde(x_range)
            ax.plot(x_range, density, color=color, lw=2)
            ax.fill_between(x_range, density, alpha=0.40, color=color, label=label)
        else:
            ax.axvline(data[0] if len(data) == 1 else 0.5, color=color,
                       linestyle="--", lw=2, label=label)

    _plot_class(genuine_scores, "#4c72b0", f"Genuine (n={len(genuine_scores)})")
    _plot_class(spoof_scores,   "#c44e52", f"Spoofed (n={len(spoof_scores)})")

    # EER threshold marker
    eer, thr = compute_eer(labels, scores)
    ax.axvline(thr, color="black", linestyle="--", linewidth=1.2,
               label=f"EER threshold ≈ {thr:.2f}")
    ax.set_xlabel("Predicted Spoof Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — Genuine vs. Spoofed  (EER={eer*100:.1f}%)")
    ax.legend()
    fig.tight_layout()
    out_path = fig_dir / "score_distribution.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[Figure] {out_path}")


def plot_det_curve(eval_result: dict, fig_dir: Path) -> None:
    """3. det_curve.png — DET curve with EER marker + ROC subplot."""
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")

    scores = eval_result["scores"]
    labels = eval_result["labels"]

    far, frr, eer_val = compute_det_curve(labels, scores)
    auc               = compute_auc(labels, scores)

    # For ROC curve: fpr / tpr
    sorted_idx    = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_idx]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    tp = fp = 0
    fpr_list: List[float] = [0.0]
    tpr_list: List[float] = [0.0]
    for lbl in sorted_labels:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        fpr_list.append(fp / max(n_neg, 1))
        tpr_list.append(tp / max(n_pos, 1))
    fpr_list.append(1.0); tpr_list.append(1.0)
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── DET curve ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(far * 100, frr * 100, color="#4c72b0", lw=2, label="DET curve")
    # EER point: closest to the diagonal
    diff    = np.abs(far - frr)
    eer_idx = int(np.argmin(diff))
    ax.scatter(far[eer_idx] * 100, frr[eer_idx] * 100,
               color="#c44e52", zorder=5, s=80,
               label=f"EER ≈ {eer_val*100:.1f}%")
    ax.plot([0, 50], [0, 50], "k--", lw=0.8, alpha=0.5, label="Equal-error line")
    ax.set_xlabel("FAR — False Alarm Rate (%)")
    ax.set_ylabel("FRR — False Reject Rate (%)")
    ax.set_title("Detection Error Tradeoff (DET) Curve")
    ax.legend()
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])

    # ── ROC curve ──────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(fpr_arr, tpr_arr, color="#55a868", lw=2,
             label=f"ROC (AUC={auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    out_path = fig_dir / "det_curve.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[Figure] {out_path}")


def plot_style_confusion_matrix(eval_result: dict, fig_dir: Path) -> None:
    """4. style_confusion_matrix.png — per-style accuracy breakdown."""
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")

    scores  = eval_result["scores"]
    labels  = eval_result["labels"]
    styles  = eval_result["styles"]

    # Threshold at EER for binary predictions
    _, thr = compute_eer(labels, scores)
    preds  = (scores >= thr).astype(np.int32)

    style_set = sorted(set(styles))

    # Build per-style confusion counts: {style: {TP, FP, TN, FN}}
    rows = []
    for style in style_set:
        mask = np.array([s == style for s in styles])
        y_true = labels[mask]
        y_pred = preds[mask]

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        spoof_acc   = tp / max(tp + fn, 1) * 100
        genuine_acc = tn / max(tn + fp, 1) * 100
        overall_acc = (tp + tn) / max(len(y_true), 1) * 100

        rows.append({
            "Style":       style,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Spoof Det. (%)":   round(spoof_acc,   1),
            "Genuine Det. (%)": round(genuine_acc, 1),
            "Overall Acc. (%)": round(overall_acc, 1),
        })

    # ── Figure: two sub-panels ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: stacked confusion bar chart
    ax = axes[0]
    x        = np.arange(len(rows))
    bar_w    = 0.2
    colors   = {"TP": "#55a868", "FP": "#c44e52", "TN": "#4c72b0", "FN": "#dd8452"}
    for i, key in enumerate(["TP", "FP", "TN", "FN"]):
        vals = [r[key] for r in rows]
        ax.bar(x + i * bar_w, vals, bar_w, label=key, color=colors[key])
    ax.set_xticks(x + 1.5 * bar_w)
    ax.set_xticklabels([r["Style"] for r in rows], fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Confusion Counts by Speaking Style")
    ax.legend()

    # Panel B: accuracy heatmap
    ax2 = axes[1]
    acc_matrix = np.array([
        [r["Spoof Det. (%)"], r["Genuine Det. (%)"], r["Overall Acc. (%)"]]
        for r in rows
    ])
    col_labels = ["Spoof Det. (%)", "Genuine Det. (%)", "Overall (%)"]
    row_labels = [r["Style"] for r in rows]
    im = ax2.imshow(acc_matrix, aspect="auto", cmap="RdYlGn",
                    vmin=0, vmax=100)
    ax2.set_xticks(range(len(col_labels)))
    ax2.set_xticklabels(col_labels, fontsize=10, rotation=20, ha="right")
    ax2.set_yticks(range(len(row_labels)))
    ax2.set_yticklabels(row_labels, fontsize=11)
    ax2.set_title("Detection Accuracy by Style (%, at EER threshold)")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax2.text(j, i, f"{acc_matrix[i, j]:.1f}",
                     ha="center", va="center", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)

    fig.tight_layout()
    out_path = fig_dir / "style_confusion_matrix.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[Figure] {out_path}")


def generate_all_figures(
    history: List[dict],
    eval_result: dict,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Figures] Saving to {fig_dir}")
    plot_training_curves(history, fig_dir)
    plot_score_distribution(eval_result, fig_dir)
    plot_det_curve(eval_result, fig_dir)
    plot_style_confusion_matrix(eval_result, fig_dir)
    print("[Figures] All 4 visualizations saved.")


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Load fold + manifest ──────────────────────────────────────────────────
    fold_path     = Path(args.fold)
    manifest_path = Path(args.manifest)

    with open(fold_path, "r") as f:
        fold = json.load(f)
    manifest = load_manifest(manifest_path)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = CSSDataset(
        manifest=manifest,
        file_list=fold["train_files"],
        root=_REPO_ROOT,
        augment=True,
        aug_snr_db=args.aug_snr_db,
    )
    val_ds = CSSDataset(
        manifest=manifest,
        file_list=fold["test_files"],
        root=_REPO_ROOT,
        augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    model = build_model(checkpoint_path, device)

    trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    print(f"[Model] Trainable params ({len(trainable)} tensors):")
    for name, shape in trainable:
        print(f"  {name}: {shape}")

    # ── Loss with class-imbalance pos_weight ──────────────────────────────────
    pos_weight = compute_pos_weight(fold, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer (only trainable params) ────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)

    # ReduceLROnPlateau: verbose kwarg removed in newer PyTorch — use manually
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # ── Output dir ───────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training with early stopping ─────────────────────────────────────────
    best_val_eer     = float("inf")
    epochs_no_improve = 0
    history: List[dict] = []
    prev_lr = args.lr

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep frozen BatchNorm layers in eval mode so their running stats
        # don't shift — they have no gradient but still track statistics.
        model.backbone.eval()

        train_loss = 0.0
        n_batches  = 0

        for waveforms, prosody, labels, _ in train_loader:
            waveforms = waveforms.to(device)
            prosody   = prosody.to(device)
            labels_f  = labels.float().to(device)

            optimizer.zero_grad()
            logits = model(waveforms, prosody, Freq_aug=False)
            loss   = criterion(logits, labels_f)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        train_loss /= max(n_batches, 1)
        val_loss, val_eer, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Detect LR reduction manually (replaces removed verbose= kwarg)
        cur_lr = optimizer.param_groups[0]["lr"]
        if cur_lr < prev_lr:
            print(f"  [LR] Reduced: {prev_lr:.2e} → {cur_lr:.2e}")
            prev_lr = cur_lr

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "val_eer":    round(val_eer,     6),
            "val_auc":    round(val_auc,     6),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"EER={val_eer*100:.2f}%  AUC={val_auc:.4f}"
        )

        if val_eer < best_val_eer:
            best_val_eer      = val_eer
            epochs_no_improve = 0
            ckpt_path = out_dir / "best_model.pt"
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_eer":          val_eer,
                    "val_auc":          val_auc,
                    "aasist_args":      AASIST_L_ARGS,
                },
                str(ckpt_path),
            )
            print(f"  ✓ Best EER={best_val_eer*100:.2f}% → {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"[Early stop] No improvement for {args.patience} epochs. "
                    f"Best EER: {best_val_eer*100:.2f}%"
                )
                break

    # ── Save history ─────────────────────────────────────────────────────────
    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Final evaluation on best checkpoint ──────────────────────────────────
    print(f"\n[Eval] Loading best checkpoint from {out_dir / 'best_model.pt'}")
    best_ckpt = torch.load(str(out_dir / "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    eval_result = evaluate_with_metadata(model, val_loader, device)
    eer_final, _ = compute_eer(eval_result["labels"], eval_result["scores"])
    auc_final    = compute_auc(eval_result["labels"], eval_result["scores"])
    print(f"[Final] EER={eer_final*100:.2f}%  AUC={auc_final:.4f}")

    # ── Visualizations ───────────────────────────────────────────────────────
    fig_dir = Path(args.fig_dir)
    generate_all_figures(history, eval_result, fig_dir)

    print(f"\n[Done] Best val EER: {best_val_eer*100:.2f}%")
    print(f"       Checkpoint:   {out_dir / 'best_model.pt'}")
    print(f"       History:      {history_path}")
    print(f"       Figures:      {fig_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-2: Train AASIST-L + prosody fusion on CSS pilot data."
    )
    p.add_argument("--fold",     required=True,
                   help="Path to fold JSON, e.g. experiment/protocols/folds/fold_00.json")
    p.add_argument("--manifest", required=True,
                   help="Path to prosody_manifest.json")
    p.add_argument("--checkpoint", default=None,
                   help="Pre-trained AASIST-L .pth path (omit = random init).")
    p.add_argument("--out_dir",  default="experiment/runs/stage2",
                   help="Directory for checkpoints and history.json.")
    p.add_argument("--fig_dir",  default="experiment/runs/figures",
                   help="Directory where the 4 PNG figures are saved.")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--patience",    type=int,   default=10,
                   help="Early stopping patience in epochs.")
    p.add_argument("--aug_snr_db",  type=float, default=15.0,
                   help="SNR (dB) for Gaussian noise augmentation on train set.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
