# ML fan shroud

Machine-learning framework for excavator cooling-fan shroud data. It turns a small table of
tested shroud geometries into a validated model that predicts air flow through each cooler
and fan power for untested designs, with an uncertainty, and suggests which runs to measure next.

> Work in progress: milestones M1–M3 (data checks, exploration, structure, models and
> validation) are in place.
> The full handover README comes with milestone M6.

## Quick start

1. Install Python 3.10 or newer, then the packages:
   ```
   pip install -r requirements.txt
   ```
2. Put the workbook `Data_for_ML_training.xlsx` into `data/raw/` (it is not stored in Git).
3. Open `fan_ml_framework.ipynb` in JupyterLab or VS Code, set `DATASET` in the first code
   cell (`"steel"`, `"plastic"` or `"ldr700"`) and run all cells.

To run every dataset at once and save executed copies to `outputs/<dataset>/`:
```
python run_all.py
```

## What is where

- `settings.py` — everything specific to a sheet: columns, units, derived features, open assumptions.
- `fanml.py` — the analysis functions the notebook calls.
- `fan_ml_framework.ipynb` — the narrative notebook.
- `guide/fan_ml_methods_guide.html` — plain-language guide to the methods (open in a browser).
- `tests/` — checks of the data layer (`python -m pytest -q`).

## Data and confidentiality

The workbook and everything derived from it (`outputs/`) are excluded from Git by `.gitignore`.
