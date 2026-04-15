"""collect_results.py — Aggregate per-fold history.json files into a summary report.

Reads every  <runs_root>/<prefix>_fold??/history.json  and the matching
best_model.pt checkpoint, then produces:

  <runs_root>/<prefix>_cv_summary.json   — machine-readable aggregate
  <runs_root>/<prefix>_cv_report.md      — human-readable Markdown table
  <runs_root>/<prefix>_cv_curves.png     — overlay of val-AUC across folds

Usage
-----
    # From SpoofingDoo_Project root:
    python3 experiment/scripts/collect_results.py \\
        --runs_root experiment/runs \\
        --prefix    stage2

    # Custom fold IDs or prefix:
    python3 experiment/scripts/collect_results.py \\
        --runs_root experiment/runs \\
        --prefix    stage3_retune \\
        --folds 00 01 02 03 04
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_fold_data(fold_dir: Path) -> Dict | None:
    """Load history.json and best_model.pt metadata for one fold directory."""
    history_path = fold_dir / "history.json"
    ckpt_path    = fold_dir / "best_model.pt"

    if not history_path.exists():
        print(f"  [Skip] {fold_dir.name}: history.json not found")
        return None

    with open(history_path) as f:
        history: List[dict] = json.load(f)

    if not history:
        print(f"  [Skip] {fold_dir.name}: history.json is empty")
        return None

    # Best epoch (by val_eer, which is what we checkpoint on)
    best = min(history, key=lambda r: r["val_eer"])

    # Read checkpoint metadata if available
    ckpt_meta: Dict = {}
    if ckpt_path.exists():
        try:
            import torch
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            ckpt_meta = {
                "ckpt_epoch": ckpt.get("epoch", best["epoch"]),
                "ckpt_eer":   ckpt.get("val_eer", best["val_eer"]),
                "ckpt_auc":   ckpt.get("val_auc", best["val_auc"]),
            }
        except Exception as e:
            print(f"  [Warn] {fold_dir.name}: could not read checkpoint — {e}")

    return {
        "fold_dir":    str(fold_dir),
        "fold_name":   fold_dir.name,
        "history":     history,
        "best_epoch":  best["epoch"],
        "best_eer":    best["val_eer"],
        "best_auc":    best["val_auc"],
        "total_epochs": len(history),
        **ckpt_meta,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Output generators
# ══════════════════════════════════════════════════════════════════════════════

def make_summary_json(folds_data: List[Dict], out_path: Path) -> Dict:
    eers = [d["best_eer"] for d in folds_data]
    aucs = [d["best_auc"] for d in folds_data]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "n_folds":       len(folds_data),
        "mean_eer":      float(np.mean(eers)),
        "std_eer":       float(np.std(eers)),
        "mean_auc":      float(np.mean(aucs)),
        "std_auc":       float(np.std(aucs)),
        "folds": [
            {
                "name":         d["fold_name"],
                "best_eer_pct": round(d["best_eer"] * 100, 2),
                "best_auc":     round(d["best_auc"], 4),
                "best_epoch":   d["best_epoch"],
                "total_epochs": d["total_epochs"],
            }
            for d in folds_data
        ],
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary] JSON → {out_path}")
    return summary


def make_markdown_report(summary: Dict, folds_data: List[Dict], out_path: Path) -> None:
    lines = [
        f"# Cross-Validation Results",
        f"",
        f"**Generated:** {summary['generated_at']}  ",
        f"**Folds:** {summary['n_folds']}",
        f"",
        "## Fold-level Results",
        "",
        "| Fold | Best EER (%) | AUC | Best Epoch / Total |",
        "|------|-------------|-----|-------------------|",
    ]
    for d in folds_data:
        lines.append(
            f"| {d['fold_name']} | {d['best_eer']*100:.2f} "
            f"| {d['best_auc']:.4f} "
            f"| {d['best_epoch']} / {d['total_epochs']} |"
        )

    lines += [
        f"| **Mean** | **{summary['mean_eer']*100:.2f}** "
        f"| **{summary['mean_auc']:.4f}** | — |",
        f"| **±Std** | **±{summary['std_eer']*100:.2f}** "
        f"| **±{summary['std_auc']:.4f}** | — |",
        "",
        "## Per-Fold Training History (last 5 epochs)",
        "",
    ]

    for d in folds_data:
        lines.append(f"### {d['fold_name']}")
        lines.append("")
        lines.append("| Epoch | Train Loss | Val Loss | EER (%) | AUC |")
        lines.append("|-------|-----------|---------|---------|-----|")
        for row in d["history"][-5:]:
            lines.append(
                f"| {row['epoch']} "
                f"| {row['train_loss']:.4f} "
                f"| {row['val_loss']:.4f} "
                f"| {row['val_eer']*100:.2f} "
                f"| {row['val_auc']:.4f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"[Summary] Markdown → {out_path}")


def make_cv_curves_png(folds_data: List[Dict], out_path: Path) -> None:
    """Overlay val-AUC and val-EER curves for all folds on two subplots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="muted")
    except ImportError:
        print("[Warn] matplotlib/seaborn not available — skipping curve plot")
        return

    palette = sns.color_palette("muted", n_colors=len(folds_data))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for idx, d in enumerate(folds_data):
        epochs    = [r["epoch"]    for r in d["history"]]
        val_auc   = [r["val_auc"] for r in d["history"]]
        val_eer   = [r["val_eer"] * 100 for r in d["history"]]
        color     = palette[idx]
        label     = d["fold_name"]

        axes[0].plot(epochs, val_auc, color=color, lw=1.5, label=label)
        axes[1].plot(epochs, val_eer, color=color, lw=1.5, label=label)

    axes[0].set_title("Validation AUC — All Folds")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AUC")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation EER (%) — All Folds")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("EER (%)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[Summary] CV curves → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-fold training history into a CV summary report."
    )
    p.add_argument("--runs_root", default="experiment/runs",
                   help="Parent directory containing fold subdirectories.")
    p.add_argument("--prefix",    default="stage2",
                   help="Common prefix of fold dirs (e.g. 'stage2' → stage2_fold00 …)")
    p.add_argument("--folds",     nargs="*", default=["00","01","02","03","04"],
                   help="Fold IDs to aggregate (default: 00 01 02 03 04).")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    root   = Path(args.runs_root)

    folds_data = []
    for fid in args.folds:
        fold_dir = root / f"{args.prefix}_fold{fid}"
        data = load_fold_data(fold_dir)
        if data:
            folds_data.append(data)

    if not folds_data:
        print("[Error] No valid fold directories found. Check --runs_root and --prefix.")
        return

    print(f"\n[Collect] Found {len(folds_data)} fold(s) under {root}")

    summary_json_path = root / f"{args.prefix}_cv_summary.json"
    report_md_path    = root / f"{args.prefix}_cv_report.md"
    curves_png_path   = root / f"{args.prefix}_cv_curves.png"

    summary = make_summary_json(folds_data, summary_json_path)
    make_markdown_report(summary, folds_data, report_md_path)
    make_cv_curves_png(folds_data, curves_png_path)

    print(f"\n{'='*56}")
    print(f"  Mean EER : {summary['mean_eer']*100:.2f}% ± {summary['std_eer']*100:.2f}%")
    print(f"  Mean AUC : {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()
