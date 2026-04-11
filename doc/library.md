# Python libraries

Third-party packages used by this repo. Install with pip (CPU or CUDA build of PyTorch as you prefer).

## Install (pip)

```text
torch
numpy
matplotlib
seaborn
librosa
soundfile
```

One line:

```bash
pip install torch numpy matplotlib seaborn librosa soundfile
```

## Where they are used

| Package | Role |
|--------|------|
| **torch** | Deep model (`AASIST_imported`, attention scripts) |
| **numpy** | Arrays and numerics everywhere |
| **matplotlib** | Plotting (`pyplot`) |
| **seaborn** | Heatmaps / styled plots in attention scripts |
| **librosa** | Audio load, features, displays (`feature_analyzer`, `pitch_extraction`) |
| **soundfile** | Reading/writing WAV in attention scripts |

## Notes

- **PyTorch**: Install the variant that matches your OS and (optional) GPU from [pytorch.org](https://pytorch.org/get-started/locally/).
- **soundfile** depends on **libsndfile** on the system. On Windows, wheels often bundle it; on Linux/macOS you may need `libsndfile` via your package manager if install fails.

Standard library only in some scripts: `json`, `os`, `pathlib`, `zipfile`, `random`, `typing` — no extra pip packages.
