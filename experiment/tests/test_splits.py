"""Protocol / split sanity checks (speaker leakage)."""

import json
from pathlib import Path


def _folds_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "protocols" / "folds"


def test_fold_files_have_no_speaker_overlap_if_present():
    folds = sorted(_folds_dir().glob("fold_*.json"))
    if not folds:
        return
    for fold_path in folds:
        data = json.loads(fold_path.read_text())
        train_spk = {str(x) for x in data.get("train_speakers", [])}
        test_spk = {str(x) for x in data.get("test_speakers", [])}
        assert not (train_spk & test_spk), f"Speaker leakage in {fold_path.name}"
