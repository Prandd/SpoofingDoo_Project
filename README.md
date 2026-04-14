# SpoofingDoo_Project

Thai spoofing-detection mini-project: **AASIST-L** + **CSS** data, optional prosody late fusion (see `doc/plan_eng.md` and `doc/structure.md`).

## Layout

- `src/` — reusable code: `models/`, `features/`, `datasets/`, `train/`, `eval/`
- `data/` — local audio, metadata, cached features (not committed; `.gitkeep` placeholders only)
- `experiment/` — configs, protocols, scripts, notebooks, tests, ablations, `runs/`
- `doc/` — plans and structure notes

## Python path

Editable install (recommended):

```bash
cd SpoofingDoo_Project
pip install -e ".[dev]"
```

Or run with `PYTHONPATH=src` (also configured for pytest in `pyproject.toml`).

## Examples

- Attention visualization: `python experiment/scripts/single_attention.py`
- Compare attention maps: `python experiment/scripts/compare_attention.py`
- Pitch demo: `python -m features.pitch_extraction` (from repo root with `PYTHONPATH=src`)

## Tests

```bash
pytest
```
