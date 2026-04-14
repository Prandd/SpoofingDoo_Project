#!/usr/bin/env python3
"""
Build speaker-disjoint 5-fold cross-validation manifests for the CSS pilot corpus.

Strategy
--------
CSS pilot (~108 files) has 6 unique speakers (spk1–spk6), split across
three speech styles:
  Formal  → spk1, spk2
  Casual  → spk3, spk4
  Excited → spk5, spk6

scikit-learn's GroupKFold(n_splits=5) is used with groups=speaker_id,
which guarantees no speaker appears in both the training and test set
of any fold.

Output — experiment/protocols/folds/fold_00.json … fold_04.json
Each JSON contains:
  fold            : int (0–4)
  n_folds         : int (5)
  train_files     : List[str]   — utterance_id (filename stem) list
  test_files      : List[str]
  train_speakers  : List[str]   — speaker IDs present in train
  test_speakers   : List[str]   — speaker IDs present in test
  train_labels    : Dict[str, int]  — utt_id → 0/1
  test_labels     : Dict[str, int]
  stats           : Dict with train/test counts (total, genuine, spoof)

Usage
-----
    # default paths
    python experiment/scripts/make_folds.py

    # custom paths
    python experiment/scripts/make_folds.py \\
        --wav_dir data/raw/wav/CSS \\
        --out_dir experiment/protocols/folds
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

_ROOT = Path(__file__).resolve().parents[2]
_N_FOLDS = 5
_LABEL_MAP: dict[str, int] = {"Bona fide": 0, "Spoofed": 1}
_STYLES = frozenset(("Formal", "Casual", "Excited"))


# ---------------------------------------------------------------------------
# Path-based helpers
# ---------------------------------------------------------------------------

def _parse_speaker(stem: str) -> str:
    """Extract 'spkN' from a CSS filename stem."""
    m = re.search(r"(spk\d+)", stem)
    return m.group(1) if m else "unknown"


def _infer_label(wav_path: Path) -> int:
    """Derive label (0=genuine, 1=spoof) from the directory hierarchy."""
    for part in wav_path.parts:
        if part in _LABEL_MAP:
            return _LABEL_MAP[part]
    raise ValueError(f"Cannot infer label from: {wav_path}")


def _infer_style(wav_path: Path) -> str:
    """Derive speech style (Formal/Casual/Excited) from the path."""
    for part in wav_path.parts:
        if part in _STYLES:
            return part
    return "Unknown"


# ---------------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------------

def build_metadata(wav_dir: Path) -> list[dict]:
    """Recursively collect {utt_id, rel_path, speaker, label, style} records."""
    records: list[dict] = []
    for wav_path in sorted(wav_dir.rglob("*.wav")):
        stem = wav_path.stem
        try:
            label = _infer_label(wav_path)
        except ValueError as exc:
            print(f"  WARN: {exc}", file=sys.stderr)
            continue
        records.append(
            {
                "utt_id": stem,
                "rel_path": str(wav_path.relative_to(_ROOT)),
                "speaker": _parse_speaker(stem),
                "label": label,
                "style": _infer_style(wav_path),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def make_folds(records: list[dict], n_folds: int = _N_FOLDS) -> list[dict]:
    """Use GroupKFold to produce speaker-disjoint folds.

    Parameters
    ----------
    records : list of dicts from build_metadata()
    n_folds : number of folds (default 5)

    Returns
    -------
    List of fold-dict, one per fold.
    """
    utt_ids = np.array([r["utt_id"] for r in records])
    labels = np.array([r["label"] for r in records])
    speakers = np.array([r["speaker"] for r in records])

    unique_spks = sorted(set(speakers.tolist()))
    n_spk = len(unique_spks)
    print(f"\nUnique speakers ({n_spk}): {unique_spks}")

    if n_spk < n_folds:
        print(
            f"  WARNING: only {n_spk} speakers for {n_folds} folds. "
            "Some folds may be small.",
            file=sys.stderr,
        )

    # Map speaker strings → integer groups for GroupKFold
    spk_to_int = {s: i for i, s in enumerate(unique_spks)}
    groups = np.array([spk_to_int[s] for s in speakers])

    gkf = GroupKFold(n_splits=n_folds)
    folds: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(utt_ids, labels, groups)
    ):
        train_ids = utt_ids[train_idx].tolist()
        test_ids = utt_ids[test_idx].tolist()
        train_spks = sorted(set(speakers[train_idx].tolist()))
        test_spks = sorted(set(speakers[test_idx].tolist()))

        # Strict speaker-disjoint check
        overlap = set(train_spks) & set(test_spks)
        if overlap:
            raise RuntimeError(
                f"Fold {fold_idx}: speaker leakage detected — {overlap}"
            )

        train_labels = {
            uid: int(lbl)
            for uid, lbl in zip(utt_ids[train_idx], labels[train_idx])
        }
        test_labels = {
            uid: int(lbl)
            for uid, lbl in zip(utt_ids[test_idx], labels[test_idx])
        }

        fold_data: dict = {
            "fold": fold_idx,
            "n_folds": n_folds,
            "train_files": train_ids,
            "test_files": test_ids,
            "train_speakers": train_spks,
            "test_speakers": test_spks,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "stats": {
                "train_total": len(train_ids),
                "train_genuine": int((labels[train_idx] == 0).sum()),
                "train_spoof": int((labels[train_idx] == 1).sum()),
                "test_total": len(test_ids),
                "test_genuine": int((labels[test_idx] == 0).sum()),
                "test_spoof": int((labels[test_idx] == 1).sum()),
            },
        }
        folds.append(fold_data)

        print(
            f"  Fold {fold_idx}: "
            f"train={len(train_ids):>3} utt ({train_spks})  |  "
            f"test={len(test_ids):>3} utt ({test_spks})"
        )

    return folds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(wav_dir: Path, out_dir: Path) -> None:
    print(f"Scanning {wav_dir} ...")
    records = build_metadata(wav_dir)

    if not records:
        print("ERROR: No .wav files found.", file=sys.stderr)
        sys.exit(1)

    # Label balance summary
    n_genuine = sum(1 for r in records if r["label"] == 0)
    n_spoof = sum(1 for r in records if r["label"] == 1)
    print(f"Total utterances: {len(records)}  "
          f"(genuine={n_genuine}, spoof={n_spoof})")

    folds = make_folds(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        out_path = out_dir / f"fold_{fold['fold']:02d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fold, f, indent=2, ensure_ascii=False)
        s = fold["stats"]
        print(
            f"  Saved {out_path.name}  "
            f"(train: {s['train_genuine']}G/{s['train_spoof']}S  |  "
            f"test: {s['test_genuine']}G/{s['test_spoof']}S)"
        )

    print(f"\nAll {len(folds)} folds saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create speaker-disjoint 5-fold CV manifests for CSS"
    )
    parser.add_argument(
        "--wav_dir",
        type=Path,
        default=_ROOT / "data" / "raw" / "wav" / "CSS",
        help="Root directory containing CSS .wav files (searched recursively).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=_ROOT / "experiment" / "protocols" / "folds",
        help="Output directory for fold_00.json … fold_04.json.",
    )
    args = parser.parse_args()
    main(args.wav_dir, args.out_dir)
