"""Dataset settings for the fan-shroud ML framework.

Everything that is specific to one sheet of the workbook lives here: column names,
units, derived features, bounds, known corrections and the open assumptions.
The analysis code in ``fanml.py`` contains no dataset-specific numbers.

When the engineers answer an open question, change the matching entry below
(and its ``status`` in ``assumptions``) and re-run the notebook.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
WORKBOOK = "Data_for_ML_training.xlsx"

# Thresholds for the data-health checks
CLOSURE_FLAG_REL = 0.005   # components vs total differ by more than 0.5 %  -> FLAG
CLOSURE_NOTE_REL = 0.0005  # ... by more than 0.05 %                        -> NOTE
FORMULA_TOL = 1e-6         # recalculated vs sheet value of a derived column
HIGH_PRECISION_DIGITS = 8  # outputs with at least this many significant digits
LOW_PRECISION_DIGITS = 5   # outputs with at most this many significant digits

# Models and validation (sections 5-6)
SELECTION_TOLERANCE = 0.05  # pick the simplest model whose LOO RMSE is within 5 % of the best
CV_INNER_FOLDS = 5          # inner folds of the nested cross-validation
GP_RESTARTS = 8             # optimiser restarts of the Gaussian process
GP_RESTARTS_INNER = 3       # ... inside the nested cross-validation (speed)
RANDOM_STATE = 0
N_JOBS = -1                 # parallel jobs for cross-validation (-1 = all cores)


# --------------------------------------------------------------------------------------
# Derived features (physics coordinates). Each takes the tidy DataFrame, returns a Series.
# --------------------------------------------------------------------------------------
def steel_immersion(d):
    """Share of the 103.8 mm fan depth inside the shroud; 77.8 mm overlap at baseline."""
    return (77.8 + d["ring"] + d["trans"]) / 103.8


def steel_split(d):
    """How the shroud length is shared between ring and transition (mm)."""
    return d["ring"] - d["trans"]


def plastic_immersion(d):
    """Fan front 573.648 mm, shroud front 532.126 mm + depth offsets, fan depth 76 mm."""
    return 1 - (573.648 - 532.126 - (d["box"] + d["ring"] + d["trans"])) / 76


def plastic_split(d):
    """Ring minus transition depth (mm), the same shape coordinate as on steel."""
    return d["ring"] - d["trans"]


DATASETS = {
    # ----------------------------------------------------------------------------------
    "steel": {
        "title": "CWL steel shroud",
        "sheet": "CWL - Stell shroud",          # spelled as in the workbook
        "header_row": 0,
        "columns": {                              # sheet column -> short name
            "Ring_Depth": "ring",
            "Transition_Depth": "trans",
            "Large Triangle": "tri_large",
            "Small Triangle": "tri_small",
            "immersion": "imm_sheet",
            "Rad_Flow_kg/s": "rad",
            "HOC_Flow_kg/s": "hoc",
            "CAC_Flow_kg/s": "cac",
            "FC_Flow_kg/s": "fc",
            "Total_Flow_kg/s": "total",
            "Fan Power - kW": "power",
        },
        "inputs": ["ring", "trans", "tri_large", "tri_small"],
        "outputs": ["rad", "hoc", "cac", "fc", "total", "power"],
        "components": ["rad", "hoc", "cac", "fc"],
        "total": "total",
        "derived": {"imm": steel_immersion, "split": steel_split},
        "check_derived": {"imm": "imm_sheet"},   # recalculated column -> sheet column
        "physics_features": ["imm", "split", "tri_large", "tri_small"],
        "targets": ["total", "power", "cac"],     # outputs modelled in sections 5-6
        "later_runs_from": 21,                    # later-runs test: train on Runs 1-20
        "key_x": "imm",
        "coverage_pairs": [("imm", "split"), ("imm", "tri_large"), ("imm", "tri_small")],
        "baseline_run": "Baseline",
        "baseline_has_inputs": False,
        "baseline_assumed_inputs": {"ring": 0.0, "trans": 0.0},   # A3; triangles unknown
        "units": {"ring": "mm", "trans": "mm", "tri_large": "-", "tri_small": "-",
                  "imm": "-", "split": "mm", "rad": "kg/s", "hoc": "kg/s", "cac": "kg/s",
                  "fc": "kg/s", "total": "kg/s", "power": "kW"},
        "labels": {"ring": "Ring depth", "trans": "Transition depth",
                   "tri_large": "Large triangle", "tri_small": "Small triangle",
                   "imm": "Immersion", "split": "Split (ring - transition)",
                   "rad": "Radiator flow", "hoc": "Hydraulic oil cooler flow",
                   "cac": "Charge-air cooler flow", "fc": "Fuel cooler flow",
                   "total": "Total flow", "power": "Fan power"},
        # Observed ranges; replace with the engineers' design limits when known (A5).
        "bounds": {"ring": (-23.0, 64.55), "trans": (-60.0, 68.1),
                   "tri_large": (0.0, 1.0), "tri_small": (0.0, 1.0)},
        "variants": {},
        "assumptions": [
            {"id": "A1", "topic": "Data origin",
             "default": "Unknown; the model learns a noise level from the data.",
             "status": "open"},
            {"id": "A3", "topic": "Steel baseline geometry",
             "default": "Zero offsets (Ring = Trans = 0); used only as an independent check.",
             "status": "open"},
            {"id": "A4", "topic": "Triangle parameters",
             "default": "Normalised 0-1; their effect is tested, not assumed.",
             "status": "open"},
            {"id": "A5", "topic": "Design limits",
             "default": "Observed min/max of each input.",
             "status": "open"},
            {"id": "A6", "topic": "Optimisation goal",
             "default": "Max total flow with CAC flow not below its baseline value (placeholder).",
             "status": "open"},
        ],
    },
    # ----------------------------------------------------------------------------------
    "plastic": {
        "title": "CWL plastic shroud",
        "sheet": "CWL - Plastic shroud",
        "header_row": 1,                          # row 0 holds merged group headers
        "columns": {
            "P1_Box_Depth": "box",
            "P3_Ring_Depth": "ring",              # the sheet orders P1, P3, P2
            "P2_Transition_Depth": "trans",
            "Rad_Mass_Flow": "rad",
            "HOC_Mass_Flow": "hoc",
            "CAC_Mass_Flow": "cac",
            "FC_Mass_Flow": "fc",
            "Total_Mass_Flow": "total",
            "Immersion": "imm_sheet",
        },
        "inputs": ["box", "ring", "trans"],
        "outputs": ["rad", "hoc", "cac", "fc", "total"],
        "components": ["rad", "hoc", "cac", "fc"],
        "total": "total",
        "derived": {"imm": plastic_immersion, "split": plastic_split},
        "check_derived": {"imm": "imm_sheet"},
        "physics_features": ["imm", "box", "ring"],
        "targets": ["total", "cac"],
        "later_runs_from": 16,                    # later-runs test: train on Runs 0-15
        "key_x": "imm",
        "coverage_pairs": [("imm", "box"), ("imm", "ring"), ("imm", "split")],
        "baseline_run": "Run 0",
        "baseline_has_inputs": True,
        "units": {"box": "mm", "ring": "mm", "trans": "mm", "imm": "-", "split": "mm",
                  "rad": "kg/s", "hoc": "kg/s", "cac": "kg/s", "fc": "kg/s", "total": "kg/s"},
        "labels": {"box": "Box depth (P1)", "ring": "Ring depth (P3)",
                   "trans": "Transition depth (P2)", "imm": "Immersion",
                   "split": "Split (ring - transition)", "rad": "Radiator flow",
                   "hoc": "Hydraulic oil cooler flow", "cac": "Charge-air cooler flow",
                   "fc": "Fuel cooler flow", "total": "Total flow"},
        "bounds": {"box": (-19.935, 58.508), "ring": (-15.0, 58.341), "trans": (-20.0, 49.602)},
        # A2: Run 17 total (3.6138) differs from its component sum (3.6739).
        "variants": {
            "corrected": {"set_total_to_component_sum": ["Run 17"]},
            "excluded": {"drop": ["Run 17"]},
        },
        "assumptions": [
            {"id": "A1", "topic": "Data origin",
             "default": "Unknown; Runs 1-15 look like simulation output, 0 and 16-21 rounded. "
                        "The model learns a noise level from the data.",
             "status": "open"},
            {"id": "A2", "topic": "Run 17 total",
             "default": "Used as recorded; 'corrected' and 'excluded' variants run as checks.",
             "status": "open"},
            {"id": "A5", "topic": "Design limits",
             "default": "Observed min/max of each input.",
             "status": "open"},
            {"id": "A6", "topic": "Optimisation goal",
             "default": "Max total flow with CAC flow not below its baseline value (placeholder).",
             "status": "open"},
        ],
    },
    # ----------------------------------------------------------------------------------
    "ldr700": {
        "title": "LDR 700",
        "sheet": "LDR 700",
        "header_row": 0,
        "columns": {
            "Parameter 1": "p1",
            "Mass_FlowRate (RAD+HOC+CAC), kg/sec": "total",
        },
        "inputs": ["p1"],
        "outputs": ["total"],
        "components": [],
        "total": "total",
        "derived": {},
        "check_derived": {},
        "physics_features": ["p1"],
        "targets": ["total"],
        "later_runs_from": None,                  # runs ordered by parameter, no later batch
        "key_x": "p1",
        "coverage_pairs": [],
        "baseline_run": "Run 2",                  # Parameter 1 = 0, assumed reference (A7)
        "baseline_has_inputs": True,
        "units": {"p1": "?", "total": "kg/s"},
        "labels": {"p1": "Parameter 1", "total": "Flow RAD+HOC+CAC"},
        "bounds": {"p1": (-0.2, 1.0)},
        "variants": {},
        "assumptions": [
            {"id": "A7", "topic": "Parameter 1",
             "default": "Meaning and unit unknown; Run 2 (Parameter 1 = 0) used as reference. "
                        "1-D demonstration only.",
             "status": "open"},
        ],
    },
}
