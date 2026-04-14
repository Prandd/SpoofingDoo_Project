#!/usr/bin/env python3
"""
Offline prosody feature extraction for the CSS pilot corpus.

Extracts per-utterance:
  - mean local jitter   (%)   — micro-pitch irregularity
  - mean local shimmer  (%)   — micro-amplitude irregularity
  - F0 standard deviation (Hz) — tonal contour variability

Uses parselmouth (Python/Praat) for jitter & shimmer; F0 is derived
from the same Praat pitch tracker so units are consistent.

Output: data/features/prosody_manifest.json
  {
    "<utt_id>": {
      "path": "data/raw/wav/...",   # relative to project root
      "jitter_mean": float,
      "shimmer_mean": float,
      "f0_std": float
    },
    ...
  }

Usage
-----
    # default paths (data/raw/wav/CSS → data/features/prosody_manifest.json)
    python experiment/scripts/extract_prosody.py

    # custom paths
    python experiment/scripts/extract_prosody.py \\
        --wav_dir data/raw/wav/CSS \\
        --out data/features/prosody_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

_ROOT = Path(__file__).resolve().parents[2]

# Thai speech fundamental frequency range (Hz).
# Lower bound captures male low-tone; upper bound captures excited female.
_F0_MIN = 75.0
_F0_MAX = 400.0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_prosody(wav_path: Path) -> dict[str, float]:
    """Extract {jitter_mean, shimmer_mean, f0_std} for one utterance.

    All analysis uses Praat defaults via parselmouth so measurements are
    directly comparable to published CSS/ASVspoof analyses.

    Returns
    -------
    dict with keys 'jitter_mean', 'shimmer_mean', 'f0_std'.
    Values are 0.0 on failure (logged as warning, not raised).
    """
    sound = parselmouth.Sound(str(wav_path))

    # ---- F0 (autocorrelation pitch tracker, Praat default) ---------------
    pitch = call(sound, "To Pitch", 0.0, _F0_MIN, _F0_MAX)
    f0_values = pitch.selected_array["frequency"]
    voiced = f0_values[f0_values > 0.0]
    f0_std = float(np.std(voiced)) if len(voiced) > 1 else 0.0

    # ---- PointProcess (periodic, cc) for jitter/shimmer ------------------
    point_process = call(sound, "To PointProcess (periodic, cc)", _F0_MIN, _F0_MAX)

    # Jitter (local): ratio of consecutive period differences to mean period.
    # Praat call signature: (0=start, 0=end, minPeriod, maxPeriod, maxJitterFactor)
    try:
        jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        jitter = float(jitter) if (jitter is not None and np.isfinite(jitter)) else 0.0
    except Exception as exc:
        warnings.warn(f"{wav_path.name}: jitter failed ({exc}); using 0.0")
        jitter = 0.0

    # Shimmer (local): ratio of consecutive amplitude differences to mean amplitude.
    # Praat call signature: (0=start, 0=end, minPeriod, maxPeriod, maxJitterFactor, maxShimmerFactor_dB)
    try:
        shimmer = call(
            [sound, point_process],
            "Get shimmer (local)",
            0.0, 0.0, 0.0001, 0.02, 1.3, 1.6,
        )
        shimmer = float(shimmer) if (shimmer is not None and np.isfinite(shimmer)) else 0.0
    except Exception as exc:
        warnings.warn(f"{wav_path.name}: shimmer failed ({exc}); using 0.0")
        shimmer = 0.0

    return {"jitter_mean": jitter, "shimmer_mean": shimmer, "f0_std": f0_std}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(wav_dir: Path, out_path: Path) -> None:
    wav_files = sorted(wav_dir.rglob("*.wav"))
    if not wav_files:
        print(f"ERROR: No .wav files found under {wav_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(wav_files)} .wav files under {wav_dir}")
    print(f"Output → {out_path}\n")

    manifest: dict = {}
    n_errors = 0

    for i, wav_path in enumerate(wav_files):
        utt_id = wav_path.stem
        try:
            feats = extract_prosody(wav_path)
        except Exception as exc:
            warnings.warn(f"SKIP {utt_id}: {exc}")
            feats = {"jitter_mean": 0.0, "shimmer_mean": 0.0, "f0_std": 0.0}
            n_errors += 1

        manifest[utt_id] = {
            "path": str(wav_path.relative_to(_ROOT)),
            **feats,
        }

        if (i + 1) % 10 == 0 or (i + 1) == len(wav_files):
            print(
                f"  [{i + 1:>3}/{len(wav_files)}] {utt_id:<55}"
                f"  jitter={feats['jitter_mean']:.4f}"
                f"  shimmer={feats['shimmer_mean']:.4f}"
                f"  f0_std={feats['f0_std']:6.2f}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(manifest)} entries saved to {out_path}.")
    if n_errors:
        print(f"  WARNING: {n_errors} file(s) failed and have 0-valued features.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract prosody features (jitter, shimmer, F0-std) from CSS .wav files"
    )
    parser.add_argument(
        "--wav_dir",
        type=Path,
        default=_ROOT / "data" / "raw" / "wav" / "CSS",
        help="Root directory containing .wav files (searched recursively).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "data" / "features" / "prosody_manifest.json",
        help="Output JSON manifest path.",
    )
    args = parser.parse_args()
    main(args.wav_dir, args.out)
