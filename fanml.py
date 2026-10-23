"""fanml - analysis helpers for the fan-shroud ML framework.

The notebook ``fan_ml_framework.ipynb`` calls these functions; every dataset-specific
number lives in ``settings.py``.

Sections
    1. Settings access, output folders and plotting style
    2. Load and validate                      (notebook section 1)
    3. Features                               (notebook section 4)
    4. Exploratory analysis                   (notebook section 2)
    5. Structure (unsupervised)               (notebook section 3)
    6. Findings text for the takeaway cells   (sections 1-4)
    7. Models                                 (notebook section 5)
    8. Validation and selection               (notebook section 6)
"""
from __future__ import annotations

import itertools
import math
import platform
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
from scipy.stats import norm, qmc
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import settings as S

# ======================================================================================
# 1. Settings access, output folders and plotting style
# ======================================================================================
INK, INK2, INK3, RULE = "#171b19", "#4b524e", "#7a817d", "#d9ddd8"
ACCENT, FLAG = "#eaa41b", "#b5441a"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DATA = SERIES[0]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

DIVERGING = LinearSegmentedColormap.from_list("fan_div", ["#e34948", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("fan_seq", ["#f4f7fb", "#9ec5f4", "#2a78d6", "#104281"])


def set_style() -> None:
    """Apply the framework's matplotlib style (quiet grid, labelled axes, fixed colours)."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.edgecolor": INK3, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": RULE, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


def get_config(name: str) -> dict:
    """Return the settings of one dataset (``steel``, ``plastic`` or ``ldr700``)."""
    if name not in S.DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Choose one of: {', '.join(S.DATASETS)}")
    cfg = dict(S.DATASETS[name])
    cfg["name"] = name
    return cfg


def label(cfg: dict, col: str) -> str:
    """Axis label with unit, e.g. 'Total flow [kg/s]'."""
    unit = cfg["units"].get(col)
    text = cfg["labels"].get(col, col)
    return f"{text} [{unit}]" if unit and unit != "-" else text


def short(cfg: dict, col: str) -> str:
    """Label without unit."""
    return cfg["labels"].get(col, col)


def output_dir(cfg: dict, sub: str = "") -> Path:
    """Folder ``outputs/<dataset>/<sub>``, created on demand."""
    path = S.OUTPUT_DIR / cfg["name"] / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig, cfg: dict, name: str) -> Path:
    """Save a figure as PNG under ``outputs/<dataset>/figures``."""
    path = output_dir(cfg, "figures") / f"{name}.png"
    fig.savefig(path)
    return path


def save_table(table: pd.DataFrame, cfg: dict, name: str) -> Path:
    """Save a table as CSV under ``outputs/<dataset>/tables``."""
    path = output_dir(cfg, "tables") / f"{name}.csv"
    table.to_csv(path)
    return path


def environment_info() -> pd.Series:
    """Versions of Python and the main libraries, printed at the top of the notebook."""
    return pd.Series({
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
    }, name="version")


def md(text: str):
    """Render Markdown in the notebook (falls back to the plain string outside Jupyter)."""
    try:
        from IPython.display import Markdown
        return Markdown(text)
    except ImportError:  # pragma: no cover
        return text


# ======================================================================================
# 2. Load and validate
# ======================================================================================
def _run_label(value) -> str:
    """Normalise the run column: 1 -> 'Run 1', 'Run 0' stays, 'Baseline' stays."""
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and float(value).is_integer()):
        return f"Run {int(value)}"
    return str(value).strip()


def _run_number(label_: str) -> int:
    match = re.search(r"(\d+)", label_)
    return int(match.group(1)) if match else 0


def load_dataset(cfg: dict, variant: str = "recorded", data_dir: Path | None = None) -> pd.DataFrame:
    """Read one sheet into a tidy table with short column names.

    Columns are matched by their header text (never by position). ``variant`` applies a
    correction defined in the settings, e.g. ``"corrected"`` or ``"excluded"`` for Run 17;
    the default ``"recorded"`` keeps the workbook values unchanged.
    """
    path = Path(data_dir or S.DATA_DIR) / S.WORKBOOK
    if not path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {path}. Put {S.WORKBOOK} into {path.parent} "
            "(it is not stored in Git).")
    raw = pd.read_excel(path, sheet_name=cfg["sheet"], header=cfg["header_row"], index_col=0)
    missing = [c for c in cfg["columns"] if c not in raw.columns]
    if missing:
        raise KeyError(f"Sheet '{cfg['sheet']}' lacks columns {missing}. "
                       f"Columns found: {list(raw.columns)}")
    df = raw[list(cfg["columns"])].rename(columns=cfg["columns"])
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = pd.Index([_run_label(i) for i in raw.index], name="run")
    df.insert(0, "run_no", [_run_number(i) for i in df.index])
    df.insert(1, "is_baseline", df.index == cfg["baseline_run"])
    df = _apply_variant(df, cfg, variant)
    df.attrs.update(dataset=cfg["name"], variant=variant)
    return df


def _apply_variant(df: pd.DataFrame, cfg: dict, variant: str) -> pd.DataFrame:
    if variant == "recorded":
        return df
    spec = cfg["variants"].get(variant)
    if spec is None:
        options = ", ".join(["recorded", *cfg["variants"]])
        raise ValueError(f"Unknown variant '{variant}' for {cfg['name']}. Options: {options}")
    df = df.copy()
    for run in spec.get("set_total_to_component_sum", []):
        df.loc[run, cfg["total"]] = df.loc[run, cfg["components"]].sum()
    return df.drop(index=spec.get("drop", []))


def design_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rows with a complete set of inputs (drops a baseline row that has no inputs)."""
    return df[df[cfg["inputs"]].notna().all(axis=1)]


def assumptions_table(cfg: dict) -> pd.DataFrame:
    """The dataset's open assumptions as a table."""
    return pd.DataFrame(cfg["assumptions"]).set_index("id")


def sig_digits(x) -> int:
    """Number of significant digits as stored, e.g. 1.5761 -> 5, 1.45239429616 -> 12."""
    if x is None or pd.isna(x):
        return 0
    s = repr(float(x)).lstrip("-").split("e")[0]
    digits = s.replace(".", "").lstrip("0").rstrip("0")
    return max(len(digits), 1)


def compress_runs(labels) -> str:
    """['Run 1', 'Run 2', 'Run 3', 'Run 17'] -> 'Runs 1-3, 17'."""
    nums = sorted(_run_number(l) for l in labels if re.search(r"\d", l))
    others = [l for l in labels if not re.search(r"\d", l)]
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}" if i == j else f"{nums[i]}–{nums[j]}")
        i = j + 1
    text = ""
    if nums:
        text = ("Run " if len(nums) == 1 else "Runs ") + ", ".join(parts)
    return ", ".join([t for t in [text, *others] if t])


@dataclass
class HealthReport:
    """Result of :func:`validate`: one row per check plus detail tables."""
    dataset: str
    variant: str
    checks: pd.DataFrame
    details: dict = field(default_factory=dict)

    @property
    def flagged_runs(self) -> list:
        return list(self.details.get("closure_flag_runs", []))

    def to_markdown(self) -> str:
        lines = [f"# Data-health report: {self.dataset} ({self.variant})", "",
                 "| Check | Status | Finding |", "| --- | --- | --- |"]
        for _, r in self.checks.iterrows():
            lines.append(f"| {r['check']} | {r['status']} | {r['finding']} |")
        return "\n".join(lines) + "\n"

    def save(self, cfg: dict) -> Path:
        path = output_dir(cfg) / "data_health.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        self.checks.to_csv(output_dir(cfg, "tables") / "data_health.csv", index=False)
        if "closure" in self.details:
            self.details["closure"].to_csv(output_dir(cfg, "tables") / "closure_check.csv")
        return path


def validate(df: pd.DataFrame, cfg: dict) -> HealthReport:
    """Run the data-health checks and return a :class:`HealthReport`.

    Status: OK = nothing found, NOTE = worth knowing, FLAG = affects numbers.
    """
    rows, details = [], {}
    inputs, outputs, comps, total = cfg["inputs"], cfg["outputs"], cfg["components"], cfg["total"]
    des = design_rows(df, cfg)

    # 1. Missing values
    no_inputs = [r for r in df.index if df.loc[r, inputs].isna().any()]
    expected = [] if cfg["baseline_has_inputs"] else [cfg["baseline_run"]]
    unexpected = [r for r in no_inputs if r not in expected]
    n_missing_out = int(df[outputs].isna().sum().sum())
    if unexpected or n_missing_out:
        rows.append(("Missing values", "FLAG",
                     f"Inputs missing in {compress_runs(unexpected) or 'no run'}; "
                     f"{n_missing_out} missing output values."))
    elif expected and expected[0] in df.index:
        rows.append(("Missing values", "NOTE",
                     f"Only the {expected[0]} row has no inputs (expected; geometry assumed, see A3)."))
    else:
        rows.append(("Missing values", "OK", "No missing inputs or outputs."))

    # 2. Components add up to the total
    if comps:
        comp_sum = df[comps].sum(axis=1)
        diff = comp_sum - df[total]
        rel = (diff / df[total]).abs()
        table = pd.DataFrame({"total": df[total], "component_sum": comp_sum,
                              "difference": diff, "difference_pct": 100 * rel})
        details["closure"] = table.sort_values("difference_pct", ascending=False)
        flag = rel[rel > S.CLOSURE_FLAG_REL].index.tolist()
        note = rel[(rel > S.CLOSURE_NOTE_REL) & (rel <= S.CLOSURE_FLAG_REL)].index.tolist()
        details["closure_flag_runs"] = flag
        parts = [f"{r}: total {df.loc[r, total]:.4f} vs component sum {comp_sum[r]:.4f} "
                 f"kg/s ({100 * rel[r]:.1f} %)" for r in flag]
        if note:
            parts.append(f"{compress_runs(note)} differ by {100 * rel[note].min():.2f}"
                         f"–{100 * rel[note].max():.2f} %")
        status = "FLAG" if flag else ("NOTE" if note else "OK")
        rows.append(("Components add up to the total", status,
                     "; ".join(parts) + "." if parts else
                     f"All runs within {100 * S.CLOSURE_NOTE_REL:.2f} %."))
    else:
        rows.append(("Components add up to the total", "n/a", "No component flows in this sheet."))

    # 3. Derived columns recalculate
    for new, sheet_col in cfg["check_derived"].items():
        recalc = cfg["derived"][new](df)
        both = recalc.notna() & df[sheet_col].notna()
        err = (recalc[both] - df.loc[both, sheet_col]).abs()
        bad = err[err > S.FORMULA_TOL].index.tolist()
        details[f"formula_{new}"] = err
        rows.append((f"Sheet column '{sheet_col}' recalculates", "FLAG" if bad else "OK",
                     f"Mismatch in {compress_runs(bad)}." if bad else
                     f"{int(both.sum())} values match the formula (max difference {err.max():.1e})."))

    # 4. Repeated values at the edges of the tested range (optimiser bounds)
    hits = []
    for c in inputs:
        v = des[c]
        for name, edge in (("min", v.min()), ("max", v.max())):
            n = int((v - edge).abs().lt(1e-9).sum())
            if n >= 2:
                hits.append(f"{short(cfg, c)} = {edge:g} ({n} runs, {name})")
    rows.append(("Repeated values at range edges", "NOTE" if hits else "OK",
                 "; ".join(hits) + ". Typical of optimiser-chosen runs." if hits else
                 "No input value repeats at its minimum or maximum."))

    # 5. Duplicate designs
    dup = des[des.duplicated(subset=inputs, keep=False)].index.tolist()
    rows.append(("Duplicate designs", "FLAG" if dup else "OK",
                 f"Same inputs in {compress_runs(dup)}." if dup else "No two runs share all inputs."))

    # 6. Number precision as a hint of the data source
    cols = comps or outputs
    digits = df[cols].apply(lambda col: col.map(sig_digits)).max(axis=1)
    details["digits"] = digits
    high = digits[digits >= S.HIGH_PRECISION_DIGITS].index.tolist()
    low = digits[digits <= S.LOW_PRECISION_DIGITS].index.tolist()
    if high and low:
        lo_d, hi_d = digits[high].min(), digits[high].max()
        span = f"{lo_d}" if lo_d == hi_d else f"{lo_d}–{hi_d}"
        rows.append(("Number precision (source hint)", "NOTE",
                     f"{compress_runs(high)} carry {span} significant digits, "
                     f"{compress_runs(low)} at most {digits[low].max()}: "
                     "possibly different sources (see A1)."))
    else:
        rows.append(("Number precision (source hint)", "OK",
                     f"Uniform precision ({digits.min()}–{digits.max()} significant digits)."))

    # 7. Correlated inputs
    if len(inputs) >= 2:
        corr = des[inputs].corr()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
        a, b = upper.abs().idxmax()
        r = upper[(a, b)]
        rows.append(("Correlated inputs", "NOTE" if abs(r) > 0.5 else "OK",
                     f"Strongest pair: {short(cfg, a)} vs {short(cfg, b)}, r = {r:.2f}."))

    # 8. Open assumptions
    open_ = [a["id"] for a in cfg["assumptions"] if a["status"] == "open"]
    rows.append(("Open assumptions", "NOTE" if open_ else "OK",
                 f"{', '.join(open_)} still open (see the assumptions table)." if open_ else
                 "All assumptions confirmed."))

    checks = pd.DataFrame(rows, columns=["check", "status", "finding"])
    return HealthReport(cfg["name"], df.attrs.get("variant", "recorded"), checks, details)


# ======================================================================================
# 3. Features
# ======================================================================================
def add_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add physics coordinates, flow fractions and baseline-normalised outputs.

    - derived features from the settings (immersion, split),
    - ``frac_<cooler>``: each cooler's share of the summed component flow (sums to 1),
    - ``<output>_rel``: each output divided by the baseline run's value.
    """
    out = df.copy()
    for name, fn in cfg["derived"].items():
        out[name] = fn(out)
    if cfg["components"]:
        comp_sum = out[cfg["components"]].sum(axis=1)
        for c in cfg["components"]:
            out[f"frac_{c}"] = out[c] / comp_sum
    if cfg["baseline_run"] in out.index:
        base = out.loc[cfg["baseline_run"]]
        for o in cfg["outputs"]:
            out[f"{o}_rel"] = out[o] / base[o]
    return out


def baseline_point(df: pd.DataFrame, cfg: dict) -> pd.Series | None:
    """The baseline row with its geometry; a row without inputs gets zero offsets (A3)."""
    if cfg["baseline_run"] not in df.index:
        return None
    row = df.loc[cfg["baseline_run"]].copy()
    if not cfg["baseline_has_inputs"]:
        for c, value in cfg.get("baseline_assumed_inputs", {}).items():
            if pd.isna(row[c]):
                row[c] = value
        frame = row.to_frame().T.apply(pd.to_numeric, errors="coerce")
        for name, fn in cfg["derived"].items():
            row[name] = float(fn(frame).iloc[0])
    return row


def feature_sets(cfg: dict) -> dict:
    """Feature sets compared in section 6: raw sheet inputs vs physics coordinates."""
    return {"raw": list(cfg["inputs"]), "physics": list(cfg["physics_features"])}


def collinearity_check(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Show why immersion can't sit next to all raw depths, and what the physics set fixes.

    For each candidate feature set: the largest absolute correlation between two features
    and the condition number of the standardised feature matrix (very large = one feature
    is a combination of others).
    """
    des = design_rows(df, cfg)
    sets = {"raw inputs": cfg["inputs"]}
    if "imm" in df.columns:
        sets["raw inputs + immersion"] = [*cfg["inputs"], "imm"]
    sets["physics features"] = cfg["physics_features"]
    rows = []
    for name, cols in sets.items():
        X = StandardScaler().fit_transform(des[cols])
        corr = np.corrcoef(X, rowvar=False) if len(cols) > 1 else np.array([[1.0]])
        off = np.abs(corr[np.triu_indices(len(cols), 1)]) if len(cols) > 1 else np.array([0.0])
        rows.append({"feature set": name, "features": ", ".join(short(cfg, c) for c in cols),
                     "max |r| between features": round(float(off.max()), 2),
                     "condition number": float(np.linalg.cond(X))})
    table = pd.DataFrame(rows).set_index("feature set")
    if "imm" in df.columns:
        r2 = LinearRegression().fit(des[cfg["inputs"]], des["imm"]).score(des[cfg["inputs"]], des["imm"])
        table.attrs["imm_r2_from_raw"] = r2
    return table


# ======================================================================================
# 4. Exploratory analysis
# ======================================================================================
def summary_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Count, mean, spread and range of every input, derived feature and output."""
    cols = [*cfg["inputs"], *[c for c in cfg["derived"] if c in df], *cfg["outputs"]]
    des = df[cols]
    t = des.describe().T[["count", "mean", "std", "min", "max"]]
    t.insert(0, "unit", [cfg["units"].get(c, "") for c in cols])
    t.insert(0, "name", [short(cfg, c) for c in cols])
    t["range / mean (%)"] = np.where(
        t.index.isin(cfg["outputs"]), 100 * (t["max"] - t["min"]) / t["mean"].abs(), np.nan)
    t["count"] = t["count"].astype(int)
    return t.round(4)


def _annotate_runs(ax, x, y, labels, fontsize=7):
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(str(_run_number(lab)) if re.search(r"\d", lab) else lab, (xi, yi),
                    xytext=(3, 3), textcoords="offset points", fontsize=fontsize, color=INK3)


def _baseline_marker(ax, b, xcol, ycol, cfg):
    if b is None or pd.isna(b.get(xcol, np.nan)) or pd.isna(b.get(ycol, np.nan)):
        return
    note = " (A3)" if not cfg["baseline_has_inputs"] else ""
    ax.scatter([b[xcol]], [b[ycol]], s=70, facecolor="none", edgecolor=INK, linewidth=1.6,
               zorder=4, label=f"baseline{note}")


def plot_responses(df: pd.DataFrame, cfg: dict, x: str | None = None, outputs=None,
                   flagged=None, ncols: int = 3):
    """Each output against one variable (default: immersion), every run labelled."""
    x = x or cfg["key_x"]
    outputs = outputs or [*cfg["outputs"], f"{cfg['total']}_rel"]
    outputs = [o for o in outputs if o in df.columns]
    flagged = flagged or []
    des = design_rows(df, cfg)
    b = baseline_point(df, cfg)
    n = len(outputs)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    for ax, o in zip(axes.flat, outputs):
        ax.scatter(des[x], des[o], s=26, color=DATA, edgecolor="white", linewidth=0.6, zorder=3)
        _annotate_runs(ax, des[x], des[o], des.index)
        fl = [r for r in flagged if r in des.index]
        if fl:
            ax.scatter(des.loc[fl, x], des.loc[fl, o], s=110, facecolor="none", edgecolor=FLAG,
                       linewidth=1.6, zorder=4, label="flagged run")
        _baseline_marker(ax, b, x, o, cfg)
        if o.endswith("_rel"):
            ax.axhline(1.0, color=INK3, lw=0.8, ls="--")
            ax.set_ylabel(f"{short(cfg, o[:-4])} / baseline [-]")
            ax.set_title(f"{short(cfg, o[:-4])} relative to baseline")
        else:
            ax.set_ylabel(label(cfg, o))
            ax.set_title(short(cfg, o))
        ax.set_xlabel(label(cfg, x))
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="upper right", ncol=len(handles), fontsize=9)
    fig.suptitle(f"{cfg['title']}: outputs vs {short(cfg, x).lower()} (numbers = run)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_input_output_grid(df: pd.DataFrame, cfg: dict, features=None, outputs=None):
    """Small multiples: every feature (rows) against the key outputs (columns)."""
    features = features or list(dict.fromkeys([*cfg["inputs"], *cfg["derived"]]))
    outputs = outputs or [o for o in ("total", "cac", "power") if o in cfg["outputs"]]
    des = design_rows(df, cfg)
    fig, axes = plt.subplots(len(features), len(outputs),
                             figsize=(3.3 * len(outputs), 2.5 * len(features)), squeeze=False)
    for i, f in enumerate(features):
        for j, o in enumerate(outputs):
            ax = axes[i, j]
            ax.scatter(des[f], des[o], s=16, color=DATA, edgecolor="white", linewidth=0.5)
            ax.set_xlabel(label(cfg, f), fontsize=9)
            ax.set_ylabel(label(cfg, o), fontsize=9)
            ax.tick_params(labelsize=8)
    fig.suptitle(f"{cfg['title']}: each feature against the key outputs",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def correlation_table(df: pd.DataFrame, cfg: dict, features=None, outputs=None,
                      n_seeds: int = 10) -> pd.DataFrame:
    """Pearson r, Spearman rho and mutual information (MI) for every feature-output pair.

    MI is estimated with the k-nearest-neighbour method (k = 3) and averaged over
    ``n_seeds`` random seeds; with ~25 runs treat it as a ranking, not a precise value.
    """
    features = features or list(dict.fromkeys([*cfg["inputs"], *cfg["derived"]]))
    outputs = outputs or cfg["outputs"]
    des = design_rows(df, cfg)
    rows = []
    for o in outputs:
        y = des[o].to_numpy()
        for f in features:
            x = des[f].to_numpy()
            mi = np.mean([mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=3,
                                                 random_state=s)[0] for s in range(n_seeds)])
            rows.append({"feature": f, "output": o,
                         "pearson": pd.Series(x).corr(pd.Series(y)),
                         "spearman": pd.Series(x).corr(pd.Series(y), method="spearman"),
                         "mi": mi})
    return pd.DataFrame(rows)


def plot_correlations(corr: pd.DataFrame, cfg: dict):
    """Three heatmaps: Pearson, Spearman (diverging, -1..1) and MI (sequential)."""
    feats = list(dict.fromkeys(corr["feature"]))
    outs = list(dict.fromkeys(corr["output"]))
    fig, axes = plt.subplots(1, 3, figsize=(4.0 + 1.0 * len(outs) * 1.6, 0.55 * len(feats) + 1.8),
                             squeeze=False)
    specs = [("pearson", "Pearson r", DIVERGING, -1, 1),
             ("spearman", "Spearman ρ", DIVERGING, -1, 1),
             ("mi", "Mutual information [nats]", SEQUENTIAL, 0, max(corr["mi"].max(), 0.05))]
    for ax, (col, title, cmap, vmin, vmax) in zip(axes.flat, specs):
        m = corr.pivot(index="feature", columns="output", values=col).loc[feats, outs]
        ax.imshow(m.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(outs)), [short(cfg, o) for o in outs], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(feats)), [short(cfg, f) for f in feats], fontsize=8)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(m.to_numpy()):
            strong = abs(v) > 0.6 * max(abs(vmin), abs(vmax))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if strong else INK)
        ax.set_title(title)
    fig.suptitle(f"{cfg['title']}: feature-output association",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def find_gaps(values, top: int = 3) -> pd.DataFrame:
    """Largest empty intervals between tested values of one variable."""
    v = np.sort(pd.Series(values).dropna().unique())
    if len(v) < 2:
        return pd.DataFrame(columns=["start", "end", "width", "share_of_range"])
    widths = np.diff(v)
    idx = np.argsort(widths)[::-1][:top]
    span = v[-1] - v[0]
    return pd.DataFrame({"start": v[idx], "end": v[idx + 1], "width": widths[idx],
                         "share_of_range": widths[idx] / span}).reset_index(drop=True)


def plot_coverage(df: pd.DataFrame, cfg: dict):
    """Where the tests are: pairs of physics coordinates plus a strip of the key variable
    with its largest gap shaded."""
    des = design_rows(df, cfg)
    b = baseline_point(df, cfg)
    pairs = cfg["coverage_pairs"]
    n = len(pairs) + 1
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), squeeze=False)
    for ax, (xa, ya) in zip(axes.flat, pairs):
        ax.scatter(des[xa], des[ya], s=24, color=DATA, edgecolor="white", linewidth=0.6, zorder=3)
        _annotate_runs(ax, des[xa], des[ya], des.index)
        _baseline_marker(ax, b, xa, ya, cfg)
        ax.set_xlabel(label(cfg, xa))
        ax.set_ylabel(label(cfg, ya))
        ax.set_title(f"{short(cfg, ya)} vs {short(cfg, xa).lower()}")
    ax = axes.flat[-1]
    x = cfg["key_x"]
    gaps = find_gaps(des[x], top=1)
    ax.scatter(des[x], np.zeros(len(des)), s=30, color=DATA, edgecolor="white", zorder=3)
    if len(gaps):
        g = gaps.iloc[0]
        ax.axvspan(g["start"], g["end"], color=ACCENT, alpha=0.25, lw=0)
        ax.annotate(f"largest gap\n{g['start']:.3g}–{g['end']:.3g}",
                    ((g["start"] + g["end"]) / 2, 0.55), ha="center", fontsize=9, color=INK)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel(label(cfg, x))
    ax.set_title(f"Tested values of {short(cfg, x).lower()}")
    fig.suptitle(f"{cfg['title']}: coverage of the tested designs",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ======================================================================================
# 5. Structure (unsupervised)
# ======================================================================================
def pca_outputs(df: pd.DataFrame, cfg: dict) -> dict | None:
    """PCA of the standardised outputs (the total is left out: it is the sum of the parts).

    Returns explained-variance ratios and loadings, or None with fewer than two outputs.
    """
    cols = [o for o in cfg["outputs"] if o != cfg["total"]]
    if len(cols) < 2:
        return None
    data = df[cols].dropna()
    X = StandardScaler().fit_transform(data)
    pca = PCA().fit(X)
    names = [f"PC{i + 1}" for i in range(len(cols))]
    return {"evr": pd.Series(pca.explained_variance_ratio_, index=names),
            "loadings": pd.DataFrame(pca.components_.T, index=cols, columns=names),
            "columns": cols, "n": len(data)}


def plot_pca_outputs(res: dict, cfg: dict):
    """Scree plot (variance per component) and loadings of the first two components."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    evr = 100 * res["evr"]
    a1.bar(evr.index, evr.values, color=[DATA] + [INK3] * (len(evr) - 1), width=0.6)
    a1.plot(evr.index, evr.cumsum().values, color=INK, marker="o", ms=4, lw=1, label="cumulative")
    for i, v in enumerate(evr.values):
        if v > 15:
            a1.text(i, v / 2, f"{v:.0f} %", ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        else:
            a1.text(i, v + 2, f"{v:.0f} %", ha="center", fontsize=8, color=INK)
    a1.set_ylim(0, 110)
    a1.set_ylabel("Share of output variation [%]")
    a1.set_title("Variation explained by each component")
    a1.legend(loc="center right", fontsize=8)
    load = res["loadings"][["PC1", "PC2"]]
    y = np.arange(len(load))
    a2.barh(y - 0.18, load["PC1"], height=0.34, color=DATA, label="PC1")
    a2.barh(y + 0.18, load["PC2"], height=0.34, color=INK3, label="PC2")
    a2.set_yticks(y, [short(cfg, c) for c in load.index], fontsize=8)
    a2.axvline(0, color=INK3, lw=0.8)
    a2.set_xlabel("Loading [-]")
    a2.set_title("What the first two components consist of")
    a2.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    a2.invert_yaxis()
    fig.suptitle(f"{cfg['title']}: principal components of the outputs (n = {res['n']})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def input_structure(df: pd.DataFrame, cfg: dict) -> dict:
    """Correlation of the raw inputs and of the physics features (quality of the test plan)."""
    des = design_rows(df, cfg)
    raw = des[cfg["inputs"]].corr()
    phys = des[cfg["physics_features"]].corr()
    evr = None
    if len(cfg["inputs"]) >= 2:
        evr = pd.Series(PCA().fit(StandardScaler().fit_transform(des[cfg["inputs"]])).explained_variance_ratio_,
                        index=[f"PC{i + 1}" for i in range(len(cfg["inputs"]))])
    return {"raw_corr": raw, "physics_corr": phys, "raw_pca_evr": evr}


def plot_input_structure(res: dict, cfg: dict):
    """Heatmaps of input correlations: raw sheet inputs vs physics features."""
    mats = [("Raw inputs", res["raw_corr"]), ("Physics features", res["physics_corr"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, (title, m) in zip(axes, mats):
        ax.imshow(m.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
        names = [short(cfg, c) for c in m.index]
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(m.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
        ax.set_title(f"{title}: correlation r")
    fig.suptitle(f"{cfg['title']}: how independently the inputs were varied",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_fractions(df: pd.DataFrame, cfg: dict, x: str | None = None):
    """Each cooler's share of the air flow against the key variable."""
    x = x or cfg["key_x"]
    des = design_rows(df, cfg)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for k, c in enumerate(cfg["components"]):
        ax.scatter(des[x], 100 * des[f"frac_{c}"], s=24, color=SERIES[k], marker=MARKERS[k],
                   edgecolor="white", linewidth=0.5, label=short(cfg, c))
    ax.set_xlabel(label(cfg, x))
    ax.set_ylabel("Share of summed cooler flow [%]")
    ax.set_title(f"{cfg['title']}: how the air is split between coolers")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout()
    return fig


def fraction_ranges(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Min and max share of each cooler, in percent."""
    des = design_rows(df, cfg)
    rows = {short(cfg, c): (100 * des[f"frac_{c}"].min(), 100 * des[f"frac_{c}"].max())
            for c in cfg["components"]}
    t = pd.DataFrame(rows, index=["min share (%)", "max share (%)"]).T
    t["max / min"] = t["max share (%)"] / t["min share (%)"]
    return t.round(2)


# ======================================================================================
# 6. Findings text for the takeaway cells
# ======================================================================================
def _bullets(items) -> str:
    return "\n".join(f"- {i}" for i in items)


def findings_validation(report: HealthReport):
    """Markdown bullets: what the data-health checks found."""
    items = [f"**{r['status']}** {r['check']}: {r['finding']}"
             for _, r in report.checks.iterrows() if r["status"] in ("FLAG", "NOTE")]
    if not items:
        items = ["All checks passed."]
    return md(f"**Findings for this dataset ({report.dataset}, {report.variant}):**\n\n" + _bullets(items))


def findings_exploration(df: pd.DataFrame, cfg: dict, corr: pd.DataFrame):
    """Markdown bullets: strongest links, best and worst runs, largest gap."""
    total, x = cfg["total"], cfg["key_x"]
    des = design_rows(df, cfg)
    items = []
    c_tot = corr[corr["output"] == total].copy()
    c_tot["abs_rho"] = c_tot["spearman"].abs()
    top = c_tot.sort_values("abs_rho", ascending=False).iloc[0]
    items.append(f"Strongest steady trend with {short(cfg, total).lower()}: "
                 f"{short(cfg, top['feature'])} (Spearman ρ = {top['spearman']:.2f}).")
    weak = c_tot[c_tot["abs_rho"] < 0.2]
    if len(weak):
        items.append("Little or no steady trend: " + ", ".join(
            f"{short(cfg, f)} (ρ = {r:.2f})" for f, r in zip(weak["feature"], weak["spearman"])) + ".")
    mi_top = c_tot.sort_values("mi", ascending=False).iloc[0]
    if mi_top["feature"] != top["feature"]:
        items.append(f"Mutual information ranks {short(cfg, mi_top['feature'])} first: "
                     "a link that is strong but not monotonic.")
    best, worst = des[total].idxmax(), des[total].idxmin()
    rel = f"{total}_rel"
    items.append(f"Highest measured {short(cfg, total).lower()}: {best}, {des.loc[best, total]:.3f} kg/s "
                 f"({des.loc[best, rel]:.2f}× baseline) at {short(cfg, x)} = {des.loc[best, x]:.3g}. "
                 f"Lowest: {worst}, {des.loc[worst, total]:.3f} kg/s ({des.loc[worst, rel]:.2f}×) "
                 f"at {short(cfg, x)} = {des.loc[worst, x]:.3g}.")
    gaps = find_gaps(des[x], top=1)
    if len(gaps):
        g = gaps.iloc[0]
        items.append(f"Largest gap in tested {short(cfg, x).lower()}: {g['start']:.3g}–{g['end']:.3g} "
                     f"({100 * g['share_of_range']:.0f} % of the tested range) with no runs.")
    return md(f"**Findings for this dataset ({cfg['name']}):**\n\n" + _bullets(items))


def findings_structure(df: pd.DataFrame, cfg: dict, pca_res, inp_res):
    """Markdown bullets: number of independent output patterns, input correlation, splits."""
    items = []
    if pca_res is not None:
        evr = 100 * pca_res["evr"]
        items.append(f"The first principal component carries {evr.iloc[0]:.0f} % of the variation in "
                     f"{len(pca_res['columns'])} outputs; the first two {evr.iloc[:2].sum():.0f} %.")
    else:
        items.append("Only one output: no output structure to analyse.")
    for key, name in (("raw_corr", "raw inputs"), ("physics_corr", "physics features")):
        m = inp_res[key]
        if len(m) > 1:
            upper = m.where(np.triu(np.ones(m.shape, dtype=bool), k=1)).stack()
            a, b = upper.abs().idxmax()
            items.append(f"Strongest correlation among {name}: {short(cfg, a)} vs {short(cfg, b)}, "
                         f"r = {upper[(a, b)]:.2f}.")
    if cfg["components"]:
        fr = fraction_ranges(df, cfg)
        k = fr["max / min"].idxmax()
        name = re.sub(r" flow$", "", k)
        items.append(f"The {name.lower()}'s share of the air flow varies most: "
                     f"{fr.loc[k, 'min share (%)']:.1f} % to {fr.loc[k, 'max share (%)']:.1f} %.")
    return md(f"**Findings for this dataset ({cfg['name']}):**\n\n" + _bullets(items))


def findings_features(check: pd.DataFrame, cfg: dict):
    """Markdown bullets: what the physics feature set changes compared with the raw inputs."""
    items = []
    if "raw inputs + immersion" in check.index:
        cn = check.loc["raw inputs + immersion", "condition number"]
        r2 = check.attrs.get("imm_r2_from_raw", float("nan"))
        items.append(f"Immersion is an exact combination of the raw depths (straight-line fit R² = {r2:.4f}); "
                     f"adding it to them makes the feature matrix singular (condition number {cn:.1e}).")
    raw, phys = check.loc["raw inputs"], check.loc["physics features"]
    if len(cfg["inputs"]) > 1:
        items.append(f"Strongest correlation between features: {raw['max |r| between features']:.2f} (raw) vs "
                     f"{phys['max |r| between features']:.2f} (physics); condition number "
                     f"{raw['condition number']:.2f} vs {phys['condition number']:.2f}.")
    items.append("Physics features used from section 5 on: " + ", ".join(short(cfg, c) for c in cfg["physics_features"]) + ".")
    return md(f"**Findings for this dataset ({cfg['name']}):**\n\n" + _bullets(items))


# ======================================================================================
# 7. Models
# ======================================================================================
class GaussianProcessModel(BaseEstimator, RegressorMixin):
    """Gaussian process (Kriging) for small engineering datasets.

    Kernel: Matern 5/2 with one length scale per input (ARD: a very long length scale
    means the input barely matters) + a quadratic trend kernel (keeps a sensible global
    shape where runs are sparse) + white noise (learned measurement/simulation noise, A1).
    Inputs are standardised; the target is normalised inside the GP. Several optimiser
    restarts guard against poor local optima, which are common with few runs.
    """

    def __init__(self, restarts: int = 8, random_state: int = 0):
        self.restarts = restarts
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3), nu=2.5)
                  + ConstantKernel(1.0) * DotProduct(1.0) ** 2
                  + WhiteKernel(1e-3, (1e-6, 1.0)))
        self.gp_ = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=self.restarts,
                                            random_state=self.random_state)
        self.Xs_ = self.scaler_.transform(X)
        self.y_ = np.asarray(y, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp_.fit(self.Xs_, self.y_)
        return self

    def conditioned(self, X_new, y_new) -> "GaussianProcessModel":
        """Copy with extra (pseudo-)observations and the kernel kept fixed.

        Used by the next-run recommender ("kriging believer"): after picking a run, its
        predicted value is treated as measured so the next pick goes somewhere else.
        """
        new = GaussianProcessModel(restarts=0, random_state=self.random_state)
        new.scaler_ = self.scaler_
        new.Xs_ = np.vstack([self.Xs_, self.scaler_.transform(np.atleast_2d(np.asarray(X_new, float)))])
        new.y_ = np.concatenate([self.y_, np.atleast_1d(np.asarray(y_new, float))])
        new.gp_ = GaussianProcessRegressor(self.gp_.kernel_, optimizer=None, normalize_y=True)
        new.gp_.fit(new.Xs_, new.y_)
        return new

    def predict(self, X, return_std: bool = False):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.gp_.predict(Xs, return_std=return_std)

    def length_scales(self) -> np.ndarray:
        """Fitted ARD length scales (in standardised input units)."""
        return np.atleast_1d(self.gp_.kernel_.k1.k1.k2.length_scale)

    def noise_share(self) -> float:
        """Learned noise variance as a share of the target variance (0 = interpolates)."""
        return float(self.gp_.kernel_.k2.noise_level)


MODEL_ORDER = ["Linear", "Quadratic", "Gaussian process", "Random forest"]   # simple -> complex
CONTRAST_ONLY = {"Random forest"}   # reported for contrast, never selected


def model_zoo(restarts: int | None = None) -> dict:
    """Model factories (each call returns a fresh, unfitted model), ordered simple -> complex."""
    r = S.GP_RESTARTS if restarts is None else restarts
    return {
        "Linear": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "Quadratic": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                           RidgeCV(alphas=np.logspace(-4, 3, 30))),
        "Gaussian process": lambda: GaussianProcessModel(restarts=r, random_state=S.RANDOM_STATE),
        "Random forest": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                                       random_state=S.RANDOM_STATE),
    }


def describe_models() -> pd.DataFrame:
    """One line per model for the notebook: what it is and its role."""
    rows = [
        ("Linear", "Straight-line fit on standardised features.", "Reference every model must beat."),
        ("Quadratic", "Second-order response surface (all squares and interactions), ridge-regularised; "
                      "the penalty is chosen by internal cross-validation.", "Candidate."),
        ("Gaussian process", "Kriging: Matern 5/2 kernel with one length scale per feature, plus a quadratic "
                             "trend and learned noise. Returns an uncertainty with every prediction.", "Candidate."),
        ("Random forest", "500 regression trees, at least 2 runs per leaf.",
         "Contrast only: step-shaped, cannot extrapolate."),
    ]
    return pd.DataFrame(rows, columns=["model", "what it is", "role"]).set_index("model")


def _fit_predict(factory, X, y, train, test):
    m = factory().fit(X[train], y[train])
    if isinstance(m, GaussianProcessModel):
        mu, sd = m.predict(X[test], return_std=True)
        return mu, sd
    return m.predict(X[test]), np.full(len(test), np.nan)


def loo_predict(factory, X, y, n_jobs: int | None = None):
    """Leave-one-out predictions (and standard deviations for the Gaussian process)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y)
    folds = [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]
    out = Parallel(n_jobs=S.N_JOBS if n_jobs is None else n_jobs)(
        delayed(_fit_predict)(factory, X, y, tr, te) for tr, te in folds)
    pred = np.array([o[0][0] for o in out])
    std = np.array([o[1][0] for o in out])
    return pred, std


def scores(y, pred, scale: float | None = None) -> dict:
    """Three views of the error on unseen runs.

    - ``q2``: 1 - PRESS / TSS, the R2 of predictions on runs left out of training.
      1 = perfect, 0 = no better than the mean, < 0 = worse. Use it to compare models
      on the same output; it depends on how much that output varies.
    - ``rmse``: typical error in the output's unit (also ``rmse_pct``, % of `scale`,
      normally the baseline value). Use it to judge whether a model is good enough.
    - ``max_abs_err``: the worst single miss (also ``max_abs_err_pct``). Use it to see
      whether one run dominates the error.
    ``mae`` (mean absolute error) is reported too. Q2 ~ 1 - (RMSE / SD of the output)^2.
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    out = {"q2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
           "rmse": rmse, "mae": float(np.mean(np.abs(err))), "max_abs_err": float(np.max(np.abs(err)))}
    for k in ("rmse", "mae", "max_abs_err"):
        out[f"{k}_pct"] = 100 * out[k] / abs(scale) if scale else np.nan
    return out


METRIC_LABELS = {"q2": "Q\u00b2 [-]", "rmse_pct": "RMSE [% of baseline]",
                 "max_abs_err_pct": "Max error [% of baseline]", "mae_pct": "MAE [% of baseline]"}


# ======================================================================================
# 8. Validation and selection
# ======================================================================================
@dataclass
class CVResult:
    """Leave-one-out results: one metrics row and one prediction table per
    (target, feature set, model)."""
    metrics: pd.DataFrame
    preds: dict

    def pred(self, target: str, feature_set: str, model: str) -> pd.DataFrame:
        return self.preds[(target, feature_set, model)]


def baseline_value(data: pd.DataFrame, cfg: dict, target: str) -> float | None:
    b = cfg["baseline_run"]
    return float(data.loc[b, target]) if b in data.index else None


def cross_validate(data: pd.DataFrame, cfg: dict, targets=None, feature_sets=None,
                   models=None, raw_targets=None) -> CVResult:
    """Leave-one-out predictions of every model on every target.

    Physics features are evaluated for all targets; the raw-input comparison only for
    ``raw_targets`` (default: the key targets), which keeps the run time down.
    """
    targets = targets or cfg["targets"]
    raw_targets = cfg.get("key_targets", targets) if raw_targets is None else raw_targets
    feature_sets = feature_sets or feature_sets_for(cfg)
    zoo = model_zoo()
    models = models or list(zoo)
    des = design_rows(data, cfg)
    rows, preds = [], {}
    for t in targets:
        y = des[t].to_numpy(float)
        scale = baseline_value(data, cfg, t)
        for fs_name, cols in feature_sets.items():
            if fs_name != "physics" and t not in raw_targets:
                continue
            X = des[cols].to_numpy(float)
            for m in models:
                p, s = loo_predict(zoo[m], X, y)
                table = pd.DataFrame({"measured": y, "predicted": p, "std": s}, index=des.index)
                table["error"] = table["predicted"] - table["measured"]
                preds[(t, fs_name, m)] = table
                sc = scores(y, p, scale)
                worst = table["error"].abs().idxmax()
                rows.append({"target": t, "features": fs_name, "model": m, **sc, "worst_run": worst})
    return CVResult(pd.DataFrame(rows), preds)


def feature_sets_for(cfg: dict) -> dict:
    """Raw vs physics feature sets; identical sets (one input) are reported once."""
    fs = feature_sets(cfg)
    if fs["raw"] == fs["physics"]:
        return {"physics": fs["physics"]}
    return fs


def _choose(rmse: dict, tol: float) -> tuple[str, str]:
    """Simplest candidate model whose RMSE is within `tol` of the best one."""
    cands = {m: v for m, v in rmse.items() if m not in CONTRAST_ONLY}
    best = min(cands, key=cands.get)
    for m in MODEL_ORDER:
        if m in cands and cands[m] <= cands[best] * (1 + tol):
            return m, best
    return best, best


def select_models(cv: CVResult, cfg: dict, feature_set: str = "physics",
                  tol: float | None = None) -> pd.DataFrame:
    """Chosen model per target (simplest within tolerance of the lowest LOO RMSE)."""
    tol = S.SELECTION_TOLERANCE if tol is None else tol
    rows = []
    for t, g in cv.metrics[cv.metrics["features"] == feature_set].groupby("target", sort=False):
        g = g.set_index("model")
        chosen, best = _choose(g["rmse"].to_dict(), tol)
        reason = ("lowest RMSE" if chosen == best else
                  f"within {100 * tol:.0f} % of {best} (RMSE {g.loc[best, 'rmse']:.4g}), simpler")
        rows.append({"target": t, "model": chosen, "features": feature_set,
                     "q2": g.loc[chosen, "q2"], "rmse": g.loc[chosen, "rmse"],
                     "rmse_pct": g.loc[chosen, "rmse_pct"],
                     "max_abs_err_pct": g.loc[chosen, "max_abs_err_pct"], "reason": reason})
    return pd.DataFrame(rows).set_index("target")


def _nested_fold(zoo, X, y, train, test, inner_folds, tol):
    k = min(inner_folds, len(train))
    kf = KFold(n_splits=k, shuffle=True, random_state=S.RANDOM_STATE)
    rmse = {}
    for m, factory in zoo.items():
        if m in CONTRAST_ONLY:
            continue
        err = []
        for itr, ite in kf.split(train):
            p, _ = _fit_predict(factory, X, y, train[itr], train[ite])
            err.append(p - y[train[ite]])
        rmse[m] = float(np.sqrt(np.mean(np.concatenate(err) ** 2)))
    chosen, _ = _choose(rmse, tol)
    p, _ = _fit_predict(zoo[chosen], X, y, train, test)
    return p[0], chosen


def nested_cv(data: pd.DataFrame, cfg: dict, targets=None, feature_set: str = "physics",
              tol: float | None = None) -> pd.DataFrame:
    """Honest score of the whole procedure 'compare models, pick one, predict'.

    Outer loop: leave one run out. Inner loop: k-fold cross-validation on the remaining
    runs picks a model with the same rule as `select_models`; that model then predicts
    the left-out run. The score therefore includes the luck of the model choice.
    """
    tol = S.SELECTION_TOLERANCE if tol is None else tol
    targets = targets or cfg.get("key_targets", cfg["targets"])
    zoo = model_zoo(restarts=S.GP_RESTARTS_INNER)
    des = design_rows(data, cfg)
    X = des[feature_sets(cfg)[feature_set]].to_numpy(float)
    n = len(des)
    rows = []
    for t in targets:
        y = des[t].to_numpy(float)
        folds = [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]
        out = Parallel(n_jobs=S.N_JOBS)(delayed(_nested_fold)(zoo, X, y, tr, te, S.CV_INNER_FOLDS, tol)
                                        for tr, te in folds)
        pred = np.array([o[0] for o in out])
        picks = pd.Series([o[1] for o in out]).value_counts()
        sc = scores(y, pred, baseline_value(data, cfg, t))
        rows.append({"target": t, "nested_q2": sc["q2"], "nested_rmse": sc["rmse"],
                     "nested_rmse_pct": sc["rmse_pct"], "nested_max_err_pct": sc["max_abs_err_pct"],
                     "models picked (of %d folds)" % n: ", ".join(f"{m} {c}" for m, c in picks.items())})
    return pd.DataFrame(rows).set_index("target")


def later_runs_test(data: pd.DataFrame, cfg: dict, targets=None, feature_set: str = "physics",
                    models=None) -> pd.DataFrame | None:
    """Train on the earlier runs, predict the later ones (as the model would be used)."""
    split = cfg.get("later_runs_from")
    if split is None:
        return None
    targets = targets or cfg.get("key_targets", cfg["targets"])
    zoo = model_zoo()
    models = models or list(zoo)
    des = design_rows(data, cfg)
    train = np.where(des["run_no"].to_numpy() < split)[0]
    test = np.where(des["run_no"].to_numpy() >= split)[0]
    rows = []
    for fs_name, cols in feature_sets_for(cfg).items():
        X = des[cols].to_numpy(float)
        for t in targets:
            y = des[t].to_numpy(float)
            for m in models:
                p, _ = _fit_predict(zoo[m], X, y, train, test)
                err = p - y[test]
                worst = des.index[test][int(np.argmax(np.abs(err)))]
                scale = baseline_value(data, cfg, t)
                rmse = float(np.sqrt(np.mean(err ** 2)))
                mx = float(np.max(np.abs(err)))
                rows.append({"target": t, "features": fs_name, "model": m, "rmse": rmse,
                             "rmse_pct": 100 * rmse / scale if scale else np.nan,
                             "max_abs_err": mx, "max_abs_err_pct": 100 * mx / scale if scale else np.nan,
                             "worst_run": worst})
    res = pd.DataFrame(rows)
    res.attrs.update(n_train=len(train), n_test=len(test),
                     train_runs=compress_runs(list(des.index[train])),
                     test_runs=compress_runs(list(des.index[test])))
    return res


def interval_coverage(cv: CVResult, target: str, feature_set: str = "physics",
                      model: str = "Gaussian process") -> dict:
    """Share of left-out runs inside the GP's 68 % and 95 % prediction bands.

    ``calibration`` is the root-mean-square of the standardised errors z = (measured -
    predicted) / std. A well-calibrated model gives about 1; 1.5 means the bands must be
    widened by 1.5 to be honest. ``inside_95_calibrated`` is the coverage after widening.
    """
    p = cv.pred(target, feature_set, model)
    z = (p["measured"] - p["predicted"]) / p["std"]
    k = float(np.sqrt(np.mean(z ** 2)))
    return {"target": target, "n": len(z), "inside_68": float((z.abs() <= 1.0).mean()),
            "inside_95": float((z.abs() <= 1.96).mean()), "calibration": k,
            "inside_95_calibrated": float((z.abs() / k <= 1.96).mean()),
            "outside_95": list(z.index[z.abs() > 1.96])}


def variant_sensitivity(cfg: dict, target: str, model: str, feature_set: str = "physics") -> pd.DataFrame | None:
    """Re-run leave-one-out under each data variant (e.g. Run 17 recorded/corrected/excluded)."""
    if not cfg["variants"]:
        return None
    rows = []
    for v in ["recorded", *cfg["variants"]]:
        d = add_features(load_dataset(cfg, variant=v), cfg)
        des = design_rows(d, cfg)
        X, y = des[feature_sets(cfg)[feature_set]].to_numpy(float), des[target].to_numpy(float)
        p, _ = loo_predict(model_zoo()[model], X, y)
        sc = scores(y, p, baseline_value(d, cfg, target))
        rows.append({"variant": v, "runs": len(des), "q2": sc["q2"], "rmse": sc["rmse"],
                     "rmse_pct": sc["rmse_pct"], "max_abs_err": sc["max_abs_err"]})
    return pd.DataFrame(rows).set_index("variant")


def baseline_check(data: pd.DataFrame, cfg: dict, selection: pd.DataFrame,
                   feature_set: str = "physics") -> pd.DataFrame | None:
    """Predict a baseline row that has no inputs (so it was never used for training).

    The baseline geometry comes from the settings (A3). Inputs that stay unknown are
    swept over their low, middle and high values; the table reports the mean and the
    range of those predictions.
    """
    if cfg["baseline_has_inputs"]:
        return None
    b = baseline_point(data, cfg)
    if b is None:
        return None
    cols = feature_sets(cfg)[feature_set]
    unknown = [c for c in cols if pd.isna(b.get(c, np.nan))]
    grid = list(itertools.product(*[(cfg["bounds"][c][0], np.mean(cfg["bounds"][c]), cfg["bounds"][c][1])
                                    for c in unknown])) or [()]
    des = design_rows(data, cfg)
    rows = []
    for t, sel in selection.iterrows():
        model = model_zoo()[sel["model"]]().fit(des[cols].to_numpy(float), des[t].to_numpy(float))
        X = np.array([[dict(zip(unknown, combo)).get(c, b[c]) for c in cols] for combo in grid], float)
        if isinstance(model, GaussianProcessModel):
            mu, sd = model.predict(X, return_std=True)
        else:
            mu, sd = model.predict(X), np.full(len(X), np.nan)
        rows.append({"target": t, "model": sel["model"], "measured": float(b[t]),
                     "predicted": float(mu.mean()), "std": float(np.nanmean(sd)) if np.isfinite(sd).any() else np.nan,
                     "range over unknown inputs": f"{mu.min():.3f} to {mu.max():.3f}",
                     "error_pct": 100 * (mu.mean() - b[t]) / b[t]})
    res = pd.DataFrame(rows).set_index("target")
    res.attrs["unknown_inputs"] = unknown
    return res


# ----- plots ---------------------------------------------------------------------------
def plot_model_comparison(cv: CVResult, cfg: dict, targets=None,
                          metrics=("q2", "rmse_pct", "max_abs_err_pct")):
    """Three error views side by side for every model and key target.

    Bars: physics features. Hollow diamonds: the same model on raw inputs. Amber: the
    model the selection rule picks (simplest within tolerance of the lowest RMSE).
    """
    m = cv.metrics
    targets = targets or [t for t in cfg.get("key_targets", cfg["targets"]) if t in set(m["target"])]
    models = [x for x in MODEL_ORDER if x in set(m["model"])]
    phys = m[m["features"] == "physics"].set_index(["target", "model"])
    raw = m[m["features"] == "raw"].set_index(["target", "model"]) if "raw" in set(m["features"]) else None
    rows = [(t, mod) for t in targets for mod in models]
    ypos, y, prev = [], 0.0, None
    for t, _ in rows:
        if prev is not None and t != prev:
            y += 0.8
        ypos.append(y)
        y += 1
        prev = t
    chosen = {t: _choose(phys.loc[t]["rmse"].to_dict(), S.SELECTION_TOLERANCE)[0] for t in targets}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics) + 2.2, 0.75 + 0.27 * y),
                             squeeze=False, sharey=True)
    for ax, met in zip(axes.flat, metrics):
        vals = np.array([phys.loc[r, met] for r in rows])
        xmax = 1.0 if met == "q2" else np.nanmax(np.concatenate(
            [vals, [raw.loc[r, met] for r in rows if raw is not None and r in raw.index]])) * 1.25
        for yy, r, v in zip(ypos, rows, vals):
            on = chosen[r[0]] == r[1]
            ax.barh(yy, max(v, 0), height=0.72, color=ACCENT if on else INK3, alpha=1 if on else 0.45)
            right = max(v, 0)
            if raw is not None and r in raw.index:
                rv = min(max(raw.loc[r, met], 0), xmax)
                ax.scatter(rv, yy, marker="D", s=22, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=4)
                right = max(right, rv)
            txt = (f"< 0 ({v:.2f})" if v < 0 else f"{v:.2f}") if met == "q2" else f"{v:.1f}"
            ax.text(right + 0.03 * xmax, yy, txt, va="center", fontsize=7.5,
                    color=INK if on else INK2, fontweight="bold" if on else "normal")
        ax.set_xlim(0, xmax * (1.2 if met == "q2" else 1.08))
        ax.set_xlabel(METRIC_LABELS[met])
        better = "higher is better" if met == "q2" else "lower is better"
        ax.set_title(f"{METRIC_LABELS[met].split(' [')[0]} ({better})")
    axes.flat[0].set_yticks(ypos, [f"{short(cfg, t)} \u00b7 {mod}" for t, mod in rows], fontsize=8)
    axes.flat[0].invert_yaxis()
    handles = [matplotlib.patches.Patch(color=ACCENT, label="chosen model (physics features)"),
               matplotlib.patches.Patch(color=INK3, alpha=0.45, label="other models (physics features)")]
    if raw is not None:
        handles.append(matplotlib.lines.Line2D([], [], marker="D", ls="", markerfacecolor="white",
                                               markeredgecolor=INK, label="same model, raw inputs"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=8.5)
    fig.suptitle(f"{cfg['title']}: accuracy on runs left out of training",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_parity(cv: CVResult, cfg: dict, target: str, feature_set: str = "physics", flagged=None,
                n_label: int = 3):
    """Measured vs leave-one-out prediction for every model (GP with 95 % bands)."""
    models = [x for x in MODEL_ORDER if (target, feature_set, x) in cv.preds]
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.7), squeeze=False)
    unit = cfg["units"].get(target, "")
    allv = pd.concat([cv.pred(target, feature_set, mo)[["measured", "predicted"]] for mo in models])
    lo, hi = allv.min().min(), allv.max().max()
    pad = 0.06 * (hi - lo)
    for ax, mo in zip(axes.flat, models):
        p = cv.pred(target, feature_set, mo)
        if p["std"].notna().any():
            ax.errorbar(p["measured"], p["predicted"], yerr=1.96 * p["std"], fmt="none",
                        ecolor=INK3, elinewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter(p["measured"], p["predicted"], s=22, color=DATA, edgecolor="white", linewidth=0.5, zorder=3)
        for r in (flagged or []):
            if r in p.index:
                ax.scatter(p.loc[r, "measured"], p.loc[r, "predicted"], s=100, facecolor="none",
                           edgecolor=FLAG, linewidth=1.5, zorder=4)
        for r in p["error"].abs().nlargest(n_label).index:
            ax.annotate(str(_run_number(r)) if re.search(r"\d", r) else r,
                        (p.loc[r, "measured"], p.loc[r, "predicted"]), xytext=(3, -9),
                        textcoords="offset points", fontsize=7, color=INK2)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK3, lw=0.9, ls="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        sc = scores(p["measured"], p["predicted"])
        ax.set_title(f"{mo}\nQ² = {sc['q2']:.2f}, RMSE = {sc['rmse']:.3g} {unit}", fontsize=10)
        ax.set_xlabel(f"Measured [{unit}]")
        ax.set_ylabel(f"Predicted, run left out [{unit}]")
    fig.suptitle(f"{cfg['title']}: {short(cfg, target).lower()}, measured vs predicted "
                 f"({feature_set} features; numbers = runs with the largest errors)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_residuals(cv: CVResult, data: pd.DataFrame, cfg: dict, target: str, model: str,
                   feature_set: str = "physics"):
    """Prediction errors of one model against each feature: a pattern means missing physics."""
    p = cv.pred(target, feature_set, model)
    cols = feature_sets(cfg)[feature_set]
    des = design_rows(data, cfg)
    rmse = scores(p["measured"], p["predicted"])["rmse"]
    unit = cfg["units"].get(target, "")
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.1), squeeze=False, sharey=True)
    for ax, c in zip(axes.flat, cols):
        ax.axhspan(-rmse, rmse, color=RULE, alpha=0.6, lw=0)
        ax.axhline(0, color=INK3, lw=0.9)
        ax.scatter(des.loc[p.index, c], p["error"], s=22, color=DATA, edgecolor="white", linewidth=0.5)
        _annotate_runs(ax, des.loc[p.index, c], p["error"], p.index)
        ax.set_xlabel(label(cfg, c))
    axes.flat[0].set_ylabel(f"Predicted - measured [{unit}]")
    fig.suptitle(f"{cfg['title']}: {model} errors on {short(cfg, target).lower()} against each feature "
                 f"(grey band = ± RMSE)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_gp_bands(cv: CVResult, data: pd.DataFrame, cfg: dict, target: str, feature_set: str = "physics"):
    """GP leave-one-out predictions with 95 % bands against the measured values, along the key variable."""
    p = cv.pred(target, feature_set, "Gaussian process")
    des = design_rows(data, cfg)
    x = des.loc[p.index, cfg["key_x"]]
    order = np.argsort(x.to_numpy())
    xs, pm, ps, ym = x.to_numpy()[order], p["predicted"].to_numpy()[order], p["std"].to_numpy()[order], p["measured"].to_numpy()[order]
    out = np.abs(ym - pm) > 1.96 * ps
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.errorbar(xs, pm, yerr=1.96 * ps, fmt="o", ms=4, color=INK3, ecolor=INK3, elinewidth=1, capsize=2,
                label="prediction, run left out (95 % band)")
    ax.scatter(xs[~out], ym[~out], s=26, color=DATA, zorder=3, label="measured, inside band")
    if out.any():
        ax.scatter(xs[out], ym[out], s=40, color=FLAG, marker="D", zorder=4, label="measured, outside band")
    ax.set_xlabel(label(cfg, cfg["key_x"]))
    ax.set_ylabel(label(cfg, target))
    ax.set_title(f"{cfg['title']}: Gaussian process uncertainty check, {short(cfg, target).lower()}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


# ----- findings -----------------------------------------------------------------------
def findings_models(cfg: dict, cv: CVResult, sel: pd.DataFrame, nested: pd.DataFrame,
                    later: pd.DataFrame | None, coverage: list, sens: pd.DataFrame | None,
                    base: pd.DataFrame | None):
    """Markdown bullets summarising sections 5-6 for the selected dataset."""
    items = []
    m = cv.metrics
    key = [t for t in cfg.get("key_targets", cfg["targets"]) if t in sel.index]
    for t in key:
        s = sel.loc[t]
        unit = cfg["units"].get(t, "")
        line = (f"**{short(cfg, t)}**: {s['model']} ({s['reason']}). Unseen runs: Q\u00b2 = {s['q2']:.2f}, "
                f"RMSE = {s['rmse']:.3g} {unit} ({s['rmse_pct']:.1f} % of baseline), "
                f"worst miss {s['max_abs_err_pct']:.1f} %.")
        if t in nested.index:
            nr = nested.loc[t]
            line += (f" With the model choice included (nested): Q\u00b2 = {nr['nested_q2']:.2f}, "
                     f"RMSE {nr['nested_rmse_pct']:.1f} %.")
        if "raw" in set(m["features"]):
            r = m[(m.target == t) & (m.features == "raw") & (m.model == s["model"])]
            if len(r):
                r = r.iloc[0]
                line += f" Same model on raw inputs: Q\u00b2 = {r['q2']:.2f}, RMSE {r['rmse_pct']:.1f} %."
        items.append(line)
    others = [t for t in sel.index if t not in key]
    if others:
        items.append("Other outputs (chosen model, unseen runs): " + "; ".join(
            f"{short(cfg, t).lower()} {sel.loc[t, 'model']}, Q\u00b2 = {sel.loc[t, 'q2']:.2f}, "
            f"RMSE {sel.loc[t, 'rmse_pct']:.1f} %" for t in others) + ".")
    if later is not None:
        sub = later[later.features == "physics"]
        parts = []
        for t in key:
            r = sub[(sub.target == t) & (sub.model == sel.loc[t, "model"])].iloc[0]
            parts.append(f"{short(cfg, t).lower()} RMSE {r['rmse_pct']:.1f} %, worst {r['max_abs_err_pct']:.1f} % "
                         f"({r['worst_run']})")
        items.append(f"Later-runs test ({later.attrs['train_runs']} \u2192 {later.attrs['test_runs']}), "
                     f"chosen models: " + "; ".join(parts) + ".")
    for c in coverage:
        verdict = ("well calibrated" if c["calibration"] <= 1.15 else
                   f"too narrow: widen by {c['calibration']:.2f}\u00d7 "
                   f"(then {100 * c['inside_95_calibrated']:.0f} % inside)")
        items.append(f"Gaussian process bands, {short(cfg, c['target']).lower()}: {100 * c['inside_95']:.0f} % "
                     f"of runs inside the 95 % band, {verdict}"
                     + (f"; outside: {compress_runs(c['outside_95'])}." if c["outside_95"] else "."))
    if sens is not None:
        u = cfg["units"].get(key[0], "")
        items.append(f"Data variants (Run 17), {short(cfg, key[0]).lower()}: " + "; ".join(
            f"{v} Q\u00b2 = {r['q2']:.2f}, RMSE {r['rmse_pct']:.1f} %, worst miss {r['max_abs_err']:.3f} {u}"
            for v, r in sens.iterrows()) + ".")
    if base is not None:
        items.append("Baseline never used in training: " + "; ".join(
            f"{short(cfg, t).lower()} predicted {r['predicted']:.3f} vs measured {r['measured']:.3f} "
            f"({r['error_pct']:+.1f} %)" for t, r in base.loc[[t for t in key if t in base.index]].iterrows())
            + " (geometry assumed, A3).")
    return md(f"**Findings for this dataset ({cfg['name']}):**\n\n" + _bullets(items))


# ======================================================================================
# 9. Interpretation (notebook section 7)
# ======================================================================================
def fit_final_models(data: pd.DataFrame, cfg: dict, selection: pd.DataFrame,
                     feature_set: str = "physics") -> dict:
    """Fit the chosen model of every target on all runs with complete inputs."""
    des = design_rows(data, cfg)
    X = des[feature_sets(cfg)[feature_set]].to_numpy(float)
    zoo = model_zoo()
    return {t: zoo[s["model"]]().fit(X, des[t].to_numpy(float)) for t, s in selection.iterrows()}


def _predict(model, X):
    """Mean and standard deviation (NaN for models without one)."""
    X = np.asarray(X, float)
    if isinstance(model, GaussianProcessModel):
        return model.predict(X, return_std=True)
    return model.predict(X), np.full(len(X), np.nan)


def physics_from_raw(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add the derived physics features to a table of raw inputs."""
    out = raw.copy()
    for name, fn in cfg["derived"].items():
        out[name] = fn(out)
    return out


def raw_from_physics(phys: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add the raw inputs implied by a table of physics features (inverse map in settings)."""
    out = phys.copy()
    if cfg.get("to_raw"):
        for k, v in cfg["to_raw"](out).items():
            out[k] = v
    for name, fn in cfg["derived"].items():
        if name not in out:
            out[name] = fn(out)
    return out


def tested_ranges(data: pd.DataFrame, cfg: dict, cols=None) -> dict:
    """Min and max of each feature over the tested runs."""
    des = design_rows(data, cfg)
    cols = cols or feature_sets(cfg)["physics"]
    return {c: (float(des[c].min()), float(des[c].max())) for c in cols}


def reference_design(data: pd.DataFrame, cfg: dict) -> pd.Series:
    """Today's design: the baseline geometry (A3), unknown inputs at mid-range."""
    b = baseline_point(data, cfg)
    raw = {c: (b[c] if b is not None and pd.notna(b.get(c, np.nan)) else float(np.mean(cfg["bounds"][c])))
           for c in cfg["inputs"]}
    return physics_from_raw(pd.DataFrame([raw]), cfg).iloc[0]


def sobol_indices(model, data: pd.DataFrame, cfg: dict, n: int | None = None) -> pd.DataFrame | None:
    """Global sensitivity of one model's prediction to each physics feature.

    S1 = share of the output variance caused by the feature alone; ST = share including
    its interactions with other features. Features are varied independently over their
    tested ranges; ``attrs['outside_limits_share']`` gives the share of those samples
    whose implied raw inputs lie outside the design limits (where the model extrapolates).
    """
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample
    cols = feature_sets(cfg)["physics"]
    if len(cols) < 2:
        return None
    rng = tested_ranges(data, cfg, cols)
    problem = {"num_vars": len(cols), "names": cols, "bounds": [list(rng[c]) for c in cols]}
    X = sobol_sample.sample(problem, n or S.SOBOL_N, calc_second_order=False, scramble=True,
                            seed=S.RANDOM_STATE)
    Y = _predict(model, X)[0]
    si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False,
                               seed=S.RANDOM_STATE)
    res = pd.DataFrame({"S1": si["S1"], "S1_conf": si["S1_conf"], "ST": si["ST"], "ST_conf": si["ST_conf"]},
                       index=cols).clip(lower=0)
    raw = raw_from_physics(pd.DataFrame(X, columns=cols), cfg)
    inside = np.ones(len(raw), bool)
    for c in cfg["inputs"]:
        lo, hi = cfg["bounds"][c]
        inside &= raw[c].between(lo - 1e-9, hi + 1e-9).to_numpy()
    res.attrs["outside_limits_share"] = float(1 - inside.mean())
    return res.sort_values("ST", ascending=False)


def plot_sobol(sobols: dict, cfg: dict):
    """Total-effect (ST) and first-order (S1) Sobol indices per key target."""
    items = [(t, s) for t, s in sobols.items() if s is not None]
    fig, axes = plt.subplots(1, len(items), figsize=(4.4 * len(items), 0.9 + 0.55 * len(items[0][1])),
                             squeeze=False, sharex=True)
    for ax, (t, sv) in zip(axes.flat, items):
        y = np.arange(len(sv))
        ax.barh(y, sv["ST"], height=0.6, color=ACCENT, alpha=0.9, label="total effect (ST)")
        ax.barh(y, sv["S1"], height=0.28, color=INK, label="alone (S1)")
        for yy, v in zip(y, sv["ST"]):
            ax.text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [short(cfg, c) for c in sv.index], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Share of output variance [-]")
        ax.set_title(short(cfg, t))
    axes.flat[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{cfg['title']}: which feature drives each output (Sobol indices over the tested ranges)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def gp_relevance(models: dict, cfg: dict) -> pd.DataFrame | None:
    """GP length scales per feature (short = matters, > 100 = switched off) for GP-chosen targets."""
    cols = feature_sets(cfg)["physics"]
    rows = {t: m.length_scales() for t, m in models.items() if isinstance(m, GaussianProcessModel)}
    if not rows:
        return None
    return pd.DataFrame(rows, index=[short(cfg, c) for c in cols]).T.rename_axis("target")


# ======================================================================================
# 10. Prediction and design (notebook sections 8-9)
# ======================================================================================
def conformal_intervals(cv: CVResult, selection: pd.DataFrame, level: float | None = None) -> pd.DataFrame:
    """Prediction intervals calibrated on the leave-one-out errors of the chosen models.

    For each target the half-width is the conformal quantile of the unseen-run errors:
    the k-th smallest of n errors with k = ceil((n + 1) * level). If the model has a
    standard deviation (Gaussian process), the errors are first divided by it
    ("normalised"), so the band is wider where the model is less sure. Otherwise the
    band has a constant width ("absolute").
    """
    level = S.INTERVAL_LEVEL if level is None else level
    rows = []
    for t, sel in selection.iterrows():
        p = cv.pred(t, sel["features"], sel["model"])
        normalised = p["std"].notna().all()
        sc = (p["error"].abs() / p["std"] if normalised else p["error"].abs()).sort_values().to_numpy()
        n = len(sc)
        k = math.ceil((n + 1) * level)
        rows.append({"target": t, "model": sel["model"], "kind": "normalised" if normalised else "absolute",
                     "level": level, "n": n, "k": min(k, n), "q": float(sc[min(k, n) - 1]),
                     "capped": k > n})
    return pd.DataFrame(rows).set_index("target")


def _with_interval(model, conf_row, X):
    mu, sd = _predict(model, X)
    half = conf_row["q"] * sd if conf_row["kind"] == "normalised" else np.full(len(mu), conf_row["q"])
    return mu, mu - half, mu + half, sd


def range_checker(data: pd.DataFrame, cfg: dict) -> dict:
    """Reference for the out-of-range warning, built from the tested runs."""
    des = design_rows(data, cfg)
    cols = cfg["range_features"]
    Z = des[cols].to_numpy(float)
    mean, std = Z.mean(axis=0), Z.std(axis=0)
    std[std == 0] = 1.0
    Zs = (Z - mean) / std
    D = cdist(Zs, Zs)
    np.fill_diagonal(D, np.inf)
    return {"cols": cols, "mean": mean, "std": std, "Zs": Zs, "runs": list(des.index),
            "threshold": float(D.min(axis=1).max()),
            "ranges": tested_ranges(data, cfg, cols)}


def range_status(designs: pd.DataFrame, cfg: dict, checker: dict) -> pd.DataFrame:
    """Status of each design: inside tested range / far from tested runs /
    outside tested range / outside design limits (in that order of severity)."""
    raw_ok = np.ones(len(designs), bool)
    for c in cfg["inputs"]:
        lo, hi = cfg["bounds"][c]
        raw_ok &= designs[c].between(lo - 1e-9, hi + 1e-9).to_numpy()
    rng_ok = np.ones(len(designs), bool)
    for c, (lo, hi) in checker["ranges"].items():
        rng_ok &= designs[c].between(lo - 1e-9, hi + 1e-9).to_numpy()
    Zs = (designs[checker["cols"]].to_numpy(float) - checker["mean"]) / checker["std"]
    D = cdist(Zs, checker["Zs"])
    nn = D.min(axis=1)
    nearest = [checker["runs"][i] for i in D.argmin(axis=1)]
    status = np.select([~raw_ok, ~rng_ok, nn > checker["threshold"]],
                       ["outside design limits", "outside tested range", "far from tested runs"],
                       "inside tested range")
    return pd.DataFrame({"status": status, "nearest_run": nearest, "distance": nn}, index=designs.index)


def predict_designs(designs: pd.DataFrame, data: pd.DataFrame, cfg: dict, models: dict,
                    conf: pd.DataFrame, checker: dict | None = None) -> pd.DataFrame:
    """What-if prediction: every modelled output with its interval, plus a range status.

    ``designs`` holds the raw inputs (one row per design, optional column 'design' as name).
    """
    d = designs.copy()
    if "design" in d.columns:
        d = d.set_index("design")
    missing = [c for c in cfg["inputs"] if c not in d.columns]
    if missing:
        raise KeyError(f"Designs lack inputs {missing}; every design needs {cfg['inputs']}")
    d = physics_from_raw(d[cfg["inputs"]].astype(float), cfg)
    checker = checker or range_checker(data, cfg)
    out = d.join(range_status(d, cfg, checker))
    X = d[feature_sets(cfg)["physics"]].to_numpy(float)
    for t, m in models.items():
        mu, lo, hi, _ = _with_interval(m, conf.loc[t], X)
        out[t], out[f"{t}_low"], out[f"{t}_high"] = mu, lo, hi
        base = baseline_value(data, cfg, t)
        if base:
            out[f"{t}_vs_baseline_pct"] = 100 * (mu / base - 1)
    comps = [c for c in cfg["components"] if c in models]
    if comps and len(comps) == len(cfg["components"]) and cfg["total"] in models:
        out["coolers_sum_minus_total"] = out[comps].sum(axis=1) - out[cfg["total"]]
    return out


def what_if_designs(cfg: dict) -> pd.DataFrame:
    """Example designs from the settings (edit them in the notebook or in settings.py)."""
    return pd.DataFrame(cfg["what_if"])


def show_predictions(pred: pd.DataFrame, cfg: dict, targets=None) -> pd.DataFrame:
    """Compact what-if table: status and 'value [low, high]' per target."""
    targets = targets or [t for t in cfg.get("key_targets", cfg["targets"]) if t in pred.columns]
    out = pd.DataFrame(index=pred.index)
    for c in [*cfg["inputs"], *[k for k in cfg["derived"]]]:
        out[short(cfg, c)] = pred[c].round(3)
    out["status"] = pred["status"]
    for t in targets:
        unit = cfg["units"].get(t, "")
        out[f"{short(cfg, t)} [{unit}]"] = [f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in
                                            zip(pred[t], pred[f"{t}_low"], pred[f"{t}_high"])]
        if f"{t}_vs_baseline_pct" in pred:
            out[f"{short(cfg, t)} vs baseline"] = [f"{v:+.1f} %" for v in pred[f"{t}_vs_baseline_pct"]]
    return out


def design_pool(data: pd.DataFrame, cfg: dict, n: int | None = None,
                keep=("inside tested range",), checker: dict | None = None) -> pd.DataFrame:
    """Candidate designs: a scrambled Sobol sample of the raw inputs within the design
    limits, with physics features and range status; only statuses in ``keep`` remain."""
    n = n or S.POOL_SIZE
    lows = np.array([cfg["bounds"][c][0] for c in cfg["inputs"]])
    highs = np.array([cfg["bounds"][c][1] for c in cfg["inputs"]])
    u = qmc.Sobol(d=len(lows), scramble=True, seed=S.RANDOM_STATE).random_base2(int(np.log2(n)))
    raw = pd.DataFrame(qmc.scale(u, lows, highs), columns=cfg["inputs"])
    pool = physics_from_raw(raw, cfg)
    pool = pool.join(range_status(pool, cfg, checker or range_checker(data, cfg)))
    return pool[pool["status"].isin(keep)].reset_index(drop=True)


def _constraint_mask(pred: pd.DataFrame, data: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, list]:
    ok = np.ones(len(pred), bool)
    text = []
    for t, op, val in cfg["optimisation"]["constraints"]:
        thr = baseline_value(data, cfg, t) if val == "baseline" else float(val)
        ok &= (pred[t] >= thr).to_numpy() if op == ">=" else (pred[t] <= thr).to_numpy()
        text.append(f"{short(cfg, t)} {op} {thr:.3f} {cfg['units'].get(t, '')}"
                    + (" (baseline)" if val == "baseline" else ""))
    return ok, text


def optimise(data: pd.DataFrame, cfg: dict, models: dict, conf: pd.DataFrame,
             checker: dict | None = None) -> pd.DataFrame:
    """Best design inside the tested range under the goal and constraints in the settings (A6).

    Two answers: the highest predicted value, and the cautious one with the highest lower
    interval bound. The best measured run that meets the constraints is added for comparison.
    """
    checker = checker or range_checker(data, cfg)
    target = cfg["optimisation"]["maximise"]
    pool = design_pool(data, cfg, checker=checker)
    pred = predict_designs(pool[cfg["inputs"]], data, cfg, models, conf, checker)
    ok, text = _constraint_mask(pred, data, cfg)
    feas = pred[ok]
    if feas.empty:
        raise RuntimeError("No candidate design meets the constraints inside the tested range.")
    rows = {"highest predicted": feas[target].idxmax(), "highest lower bound (cautious)": feas[f"{target}_low"].idxmax()}
    res = feas.loc[list(rows.values())].copy()
    res.index = list(rows)
    des = design_rows(data, cfg)
    meas_ok = np.ones(len(des), bool)
    for t, op, val in cfg["optimisation"]["constraints"]:
        thr = baseline_value(data, cfg, t) if val == "baseline" else float(val)
        meas_ok &= (des[t] >= thr).to_numpy() if op == ">=" else (des[t] <= thr).to_numpy()
    best_run = des[meas_ok][target].idxmax()
    br = des.loc[[best_run], [*cfg["inputs"], *cfg["derived"], *models]].copy()
    br.index = [f"best measured ({best_run})"]
    res = pd.concat([res, br])
    res.attrs.update(constraints=text, n_pool=len(pool), n_feasible=int(ok.sum()), target=target)
    return res


def tradeoff(data: pd.DataFrame, cfg: dict, models: dict, conf: pd.DataFrame,
             checker: dict | None = None):
    """Predicted trade-off between two outputs over the candidate designs (tested range only).

    Returns the candidate predictions and the Pareto front (designs that cannot improve one
    output without worsening the other), or None if no trade-off is defined.
    """
    spec = cfg.get("tradeoff")
    if not spec:
        return None, None
    (a, da), (b, db) = spec
    pool = design_pool(data, cfg, checker=checker)
    pred = predict_designs(pool[cfg["inputs"]], data, cfg, models, conf, checker)
    A = pred[a].to_numpy() * (1 if da == "max" else -1)
    B = pred[b].to_numpy() * (1 if db == "max" else -1)
    order = np.argsort(-A)
    front, best_b = [], -np.inf
    for i in order:
        if B[i] > best_b:
            front.append(i)
            best_b = B[i]
    return pred, pred.iloc[front].sort_values(a)


def expected_improvement(mu, sd, best):
    """Expected gain over ``best`` (maximisation) for normal predictions."""
    sd = np.maximum(sd, 1e-12)
    z = (mu - best) / sd
    return (mu - best) * norm.cdf(z) + sd * norm.pdf(z)


def _inside_hull(Z_pool: np.ndarray, Z_runs: np.ndarray) -> np.ndarray:
    """True where a candidate lies inside the convex hull of the tested runs."""
    if Z_runs.shape[1] == 1:
        return (Z_pool[:, 0] >= Z_runs[:, 0].min()) & (Z_pool[:, 0] <= Z_runs[:, 0].max())
    from scipy.spatial import Delaunay
    return Delaunay(Z_runs).find_simplex(Z_pool) >= 0


def suggest_next_runs(data: pd.DataFrame, cfg: dict, models: dict, conf: pd.DataFrame,
                      calibration: float = 1.0, n_improve: int | None = None,
                      n_explore: int | None = None, checker: dict | None = None) -> pd.DataFrame:
    """Recommend the next runs to measure.

    - "improve design": highest expected improvement (EI) of the goal output among designs
      that meet the constraints. EI uses a Gaussian process fitted for this purpose (even
      when another model was chosen for prediction), its uncertainty widened by the
      calibration factor from section 6.6. Each pick is treated as measured (kriging
      believer) and keeps a minimum distance from earlier picks, so the batch spreads out.
    - "improve model": fills the largest gap. Among designs inside the tested region (the
      convex hull of the runs in the range features), pick the one farthest from every
      tested run and every earlier pick.
    """
    n_improve = S.N_NEXT_IMPROVE if n_improve is None else n_improve
    n_explore = S.N_NEXT_EXPLORE if n_explore is None else n_explore
    checker = checker or range_checker(data, cfg)
    target = cfg["optimisation"]["maximise"]
    cols = feature_sets(cfg)["physics"]
    des = design_rows(data, cfg)
    gp = GaussianProcessModel(restarts=S.GP_RESTARTS, random_state=S.RANDOM_STATE).fit(
        des[cols].to_numpy(float), des[target].to_numpy(float))
    pool = design_pool(data, cfg, keep=("inside tested range", "far from tested runs"), checker=checker)
    pred = predict_designs(pool[cfg["inputs"]], data, cfg, models, conf, checker)
    feas, _ = _constraint_mask(pred, data, cfg)
    meas_ok = np.ones(len(des), bool)
    for t, op, val in cfg["optimisation"]["constraints"]:
        thr = baseline_value(data, cfg, t) if val == "baseline" else float(val)
        meas_ok &= (des[t] >= thr).to_numpy() if op == ">=" else (des[t] <= thr).to_numpy()
    best = float(des[meas_ok][target].max())
    X = pool[cols].to_numpy(float)
    Zp = (pool[checker["cols"]].to_numpy(float) - checker["mean"]) / checker["std"]
    hull = _inside_hull(Zp, checker["Zs"])
    min_gap = 0.5 * checker["threshold"]
    k = max(calibration, 1.0)
    picks, cur = [], gp
    for _ in range(n_improve):
        mu, sd = cur.predict(X, return_std=True)
        sd = sd * k
        score = np.where(feas, expected_improvement(mu, sd, best), -np.inf)
        if picks:
            near = cdist(Zp, Zp[[p[0] for p in picks]]).min(axis=1) < min_gap
            score[near] = -np.inf
        j = int(np.argmax(score))
        picks.append((j, "improve design", float(score[j]), float(sd[j])))
        cur = cur.conditioned(X[j:j + 1], mu[j:j + 1])
    for _ in range(n_explore):
        ref_pts = np.vstack([checker["Zs"], Zp[[p[0] for p in picks]]]) if picks else checker["Zs"]
        dist = cdist(Zp, ref_pts).min(axis=1)
        score = np.where(hull, dist, -np.inf)
        j = int(np.argmax(score))
        sd_j = float(cur.predict(X[j:j + 1], return_std=True)[1][0] * k)
        picks.append((j, "improve model", float(dist[j]), sd_j))
        cur = cur.conditioned(X[j:j + 1], cur.predict(X[j:j + 1]))
    rows = pred.iloc[[p[0] for p in picks]].copy()
    rows.insert(0, "purpose", [p[1] for p in picks])
    rows["gp_std_calibrated"] = [p[3] for p in picks]
    rows["expected_improvement"] = [p[2] if p[1] == "improve design" else np.nan for p in picks]
    rows["gap_distance"] = [p[2] if p[1] == "improve model" else np.nan for p in picks]
    rows.index = [f"N{i + 1}" for i in range(len(rows))]
    rows.attrs.update(best_measured=best, target=target, n_pool=len(pool))
    return rows


# ----- plots (sections 7-9) -------------------------------------------------------------
def _grid_status(phys: pd.DataFrame, cfg: dict, checker: dict) -> np.ndarray:
    return range_status(raw_from_physics(phys, cfg), cfg, checker)["status"].to_numpy()


def plot_effect_curves(data: pd.DataFrame, cfg: dict, models: dict, conf: pd.DataFrame,
                       ref: pd.Series, targets=None, n: int = 60, checker: dict | None = None):
    """How each output changes when one feature moves away from today's design.

    Solid line: inside the tested range; dotted: far from tested runs or outside the design
    limits. Shaded band: conformal interval. Grey dots: all runs (other features differ).
    """
    checker = checker or range_checker(data, cfg)
    targets = targets or cfg.get("key_targets", cfg["targets"])
    cols = feature_sets(cfg)["physics"]
    des = design_rows(data, cfg)
    rng = tested_ranges(data, cfg, cols)
    fig, axes = plt.subplots(len(targets), len(cols), figsize=(3.4 * len(cols), 2.7 * len(targets)),
                             squeeze=False, sharey="row")
    for i, t in enumerate(targets):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            grid = np.linspace(*rng[c], n)
            phys = pd.DataFrame([ref[cols].to_dict()] * n)
            phys[c] = grid
            mu, lo, hi, _ = _with_interval(models[t], conf.loc[t], phys[cols].to_numpy(float))
            st = _grid_status(phys, cfg, checker)
            ok = st == "inside tested range"
            ax.fill_between(grid, lo, hi, color=DATA, alpha=0.15, lw=0)
            ax.plot(grid, np.where(ok, mu, np.nan), color=DATA, lw=2)
            ax.plot(grid, np.where(~ok, mu, np.nan), color=DATA, lw=1.5, ls=":")
            ax.scatter(des[c], des[t], s=10, color=INK3, alpha=0.5, zorder=1)
            ax.axvline(ref[c], color=INK, lw=0.9, ls="--")
            if i == len(targets) - 1:
                ax.set_xlabel(label(cfg, c), fontsize=9)
            if j == 0:
                ax.set_ylabel(label(cfg, t), fontsize=9)
            ax.tick_params(labelsize=8)
    fig.suptitle(f"{cfg['title']}: effect of each feature, others at today's design (dashed line); "
                 f"band = {int(100 * conf['level'].iloc[0])} % interval, dotted = outside the tested region",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_response_map(data: pd.DataFrame, cfg: dict, models: dict, conf: pd.DataFrame, ref: pd.Series,
                      target: str, n: int = 70, checker: dict | None = None, ax=None,
                      optimum: pd.DataFrame | None = None, proposals: pd.DataFrame | None = None):
    """Predicted output over the two map features (others at today's design).

    White: outside the design limits. Hatched: far from tested runs. Dots: tested runs.
    Optional markers: the optimum (star) and proposed next runs (numbered).
    """
    pair = cfg.get("map_pair")
    if not pair:
        return None
    checker = checker or range_checker(data, cfg)
    cols = feature_sets(cfg)["physics"]
    xa, ya = pair
    rng = tested_ranges(data, cfg, [xa, ya])
    gx, gy = np.meshgrid(np.linspace(*rng[xa], n), np.linspace(*rng[ya], n))
    phys = pd.DataFrame([ref[cols].to_dict()] * gx.size)
    phys[xa], phys[ya] = gx.ravel(), gy.ravel()
    mu = _predict(models[target], phys[cols].to_numpy(float))[0].reshape(gx.shape)
    st = _grid_status(phys, cfg, checker).reshape(gx.shape)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
    else:
        fig = ax.figure
    Z = np.ma.masked_where(st == "outside design limits", mu)
    cs = ax.contourf(gx, gy, Z, levels=14, cmap=SEQUENTIAL)
    lines = ax.contour(gx, gy, Z, levels=7, colors="white", linewidths=0.6, alpha=0.8)
    ax.clabel(lines, fontsize=7, fmt="%.2f")
    far = np.ma.masked_where(st != "far from tested runs", np.ones_like(mu))
    ax.contourf(gx, gy, far, levels=[0.5, 1.5], colors="none", hatches=["////"])
    des = design_rows(data, cfg)
    ax.scatter(des[xa], des[ya], s=18, color="white", edgecolor=INK, linewidth=0.8, zorder=3, label="tested runs")
    ax.scatter([ref[xa]], [ref[ya]], s=70, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=4,
               label="today's design")
    if optimum is not None:
        o = optimum.iloc[0]
        ax.scatter([o[xa]], [o[ya]], marker="*", s=220, color=ACCENT, edgecolor=INK, zorder=5,
                   label="model optimum")
    if proposals is not None:
        for name, r in proposals.iterrows():
            mk = "^" if r["purpose"] == "improve design" else "s"
            ax.scatter([r[xa]], [r[ya]], marker=mk, s=70, color=FLAG if mk == "^" else "white",
                       edgecolor=INK, zorder=5)
            ax.annotate(name, (r[xa], r[ya]), xytext=(5, 4), textcoords="offset points", fontsize=8,
                        fontweight="bold", color=INK)
        ax.scatter([], [], marker="^", color=FLAG, edgecolor=INK, label="next run: improve design")
        ax.scatter([], [], marker="s", color="white", edgecolor=INK, label="next run: improve model")
    cb = fig.colorbar(cs, ax=ax, shrink=0.9)
    cb.set_label(f"Predicted {label(cfg, target).lower()}")
    ax.set_xlabel(label(cfg, xa))
    ax.set_ylabel(label(cfg, ya))
    note = "blank = outside design limits" + (", hatched = far from tested runs"
                                              if (st == "far from tested runs").any() else "")
    ax.set_title(f"{cfg['title']}: predicted {short(cfg, target).lower()}, other features at "
                 f"today's design ({note})", fontsize=10)
    ax.legend(fontsize=8, loc="best", framealpha=0.85, frameon=True)
    if own:
        fig.tight_layout()
    return fig


def plot_tradeoff(pred: pd.DataFrame, front: pd.DataFrame, data: pd.DataFrame, cfg: dict):
    """Candidate designs (grey), the Pareto front (amber), tested runs (blue) and today's design."""
    (a, da), (b, db) = cfg["tradeoff"]
    des = design_rows(data, cfg)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.scatter(pred[a], pred[b], s=6, color=INK3, alpha=0.25, lw=0, label="candidate designs (predicted)")
    ax.plot(front[a], front[b], color=ACCENT, lw=2, marker="o", ms=4, label="best trade-off (Pareto front)")
    ax.scatter(des[a], des[b], s=24, color=DATA, edgecolor="white", zorder=3, label="tested runs")
    if cfg["baseline_run"] in data.index:
        ax.scatter([data.loc[cfg["baseline_run"], a]], [data.loc[cfg["baseline_run"], b]], s=80,
                   facecolor="none", edgecolor=INK, linewidth=1.6, zorder=4, label="baseline")
    ax.set_xlabel(f"{label(cfg, a)} ({'higher' if da == 'max' else 'lower'} is better)")
    ax.set_ylabel(f"{label(cfg, b)} ({'higher' if db == 'max' else 'lower'} is better)")
    ax.set_title(f"{cfg['title']}: trade-off between {short(cfg, a).lower()} and {short(cfg, b).lower()}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_next_runs_1d(data: pd.DataFrame, cfg: dict, models: dict, conf: pd.DataFrame, ref: pd.Series,
                      proposals: pd.DataFrame, optimum: pd.DataFrame | None = None, n: int = 200):
    """1-D datasets: prediction with interval along the single feature, plus the proposals."""
    c = feature_sets(cfg)["physics"][0]
    t = cfg["optimisation"]["maximise"]
    lo_b, hi_b = cfg["bounds"][c]
    grid = np.linspace(lo_b, hi_b, n)
    mu, lo, hi, _ = _with_interval(models[t], conf.loc[t], grid.reshape(-1, 1))
    des = design_rows(data, cfg)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.fill_between(grid, lo, hi, color=DATA, alpha=0.15, lw=0, label=f"{int(100 * conf.loc[t, 'level'])} % interval")
    ax.plot(grid, mu, color=DATA, lw=2, label="prediction")
    ax.scatter(des[c], des[t], s=30, color=INK, zorder=3, label="tested runs")
    for name, r in proposals.iterrows():
        ax.axvline(r[c], color=FLAG if r["purpose"] == "improve design" else INK3, lw=1, ls="--")
        ax.annotate(name, (r[c], ax.get_ylim()[1]), xytext=(2, -12), textcoords="offset points", fontsize=8,
                    fontweight="bold")
    if optimum is not None:
        o = optimum.iloc[0]
        ax.scatter([o[c]], [o[t]], marker="*", s=200, color=ACCENT, edgecolor=INK, zorder=5, label="model optimum")
    ax.set_xlabel(label(cfg, c))
    ax.set_ylabel(label(cfg, t))
    ax.set_title(f"{cfg['title']}: prediction, optimum and next runs (dashed lines)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


# ----- findings (sections 7-9) ----------------------------------------------------------
def findings_interpretation(cfg: dict, sobols: dict, relevance: pd.DataFrame | None):
    items = []
    for t, sv in sobols.items():
        if sv is None:
            items.append(f"{short(cfg, t)}: only one feature, no sensitivity split.")
            continue
        parts = ", ".join(f"{short(cfg, c).lower()} {v:.2f}" for c, v in sv["ST"].items())
        items.append(f"**{short(cfg, t)}**, total-effect indices: {parts}.")
    shares = [sv.attrs["outside_limits_share"] for sv in sobols.values() if sv is not None]
    if shares:
        items.append(f"{100 * max(shares):.0f} % of the sampled feature combinations lie outside the design "
                     "limits; there the model extrapolates, so treat small indices as approximate.")
    if relevance is not None:
        off = [c for c in relevance.columns if (relevance[c] > 100).all()]
        if off:
            items.append("Gaussian process switched off (length scale > 100 for every output): "
                         + ", ".join(off) + ".")
    return md(f"**Findings for this dataset ({cfg['name']}):**\n\n" + _bullets(items))


def findings_prediction(cfg: dict, data: pd.DataFrame, conf: pd.DataFrame, pred: pd.DataFrame):
    items = []
    for t in cfg.get("key_targets", cfg["targets"]):
        r = conf.loc[t]
        base = baseline_value(data, cfg, t)
        if r["kind"] == "absolute":
            w = f"±{r['q']:.3g} {cfg['units'].get(t, '')}" + (f" ({100 * r['q'] / base:.1f} % of baseline)" if base else "")
        else:
            inside = pred["status"] == "inside tested range"
            half = (pred[f"{t}_high"] - pred[t])[inside] if inside.any() else pred[f"{t}_high"] - pred[t]
            w = (f"±{r['q']:.2f} × the GP standard deviation (for the example designs inside the tested "
                 f"range: ±{half.min():.3g} to ±{half.max():.3g} {cfg['units'].get(t, '')})")
        items.append(f"{short(cfg, t)}: {int(100 * r['level'])} % interval {w}"
                     + (" (capped at the largest error: too few runs for this level)" if r["capped"] else "") + ".")
    for name, r in pred.iterrows():
        if r["status"] != "inside tested range":
            items.append(f"'{name}' is **{r['status']}** (nearest run {r['nearest_run']}): treat its prediction as a guess.")
    return md(f"**Findings for this dataset ({cfg['name']}):**\n\n" + _bullets(items))


def findings_design(cfg: dict, data: pd.DataFrame, opt: pd.DataFrame, front: pd.DataFrame | None,
                    nxt: pd.DataFrame):
    t = opt.attrs["target"]
    unit = cfg["units"].get(t, "")
    base = baseline_value(data, cfg, t)
    items = []
    same = np.allclose(opt.iloc[0][cfg["inputs"]].to_numpy(float), opt.iloc[1][cfg["inputs"]].to_numpy(float))
    names = opt.index[:1] if same else opt.index[:2]
    if same:
        items.append("The highest and the cautious optimum are the same design (the interval has a constant width).")
    for name in names:
        r = opt.loc[name]
        geom = ", ".join(f"{short(cfg, c).lower()} {r[c]:.3g}" for c in [*cfg["inputs"], *cfg["derived"]])
        items.append(f"**{name.capitalize()}**: {short(cfg, t).lower()} {r[t]:.3f} {unit} "
                     f"[{r[f'{t}_low']:.3f}, {r[f'{t}_high']:.3f}]"
                     + (f", {100 * (r[t] / base - 1):+.1f} % vs baseline" if base else "") + f" ({geom}).")
    bm = opt.index[-1]
    items.append(f"For comparison, {bm}: {opt.loc[bm, t]:.3f} {unit}.")
    if opt.attrs["constraints"]:
        items.append("Constraints (A6, placeholder): " + "; ".join(opt.attrs["constraints"]) + ".")
    if front is not None and len(front):
        (a, da), (b, db) = cfg["tradeoff"]
        ua, ub = cfg["units"].get(a, ""), cfg["units"].get(b, "")
        items.append(f"Trade-off front: {len(front)} designs from {short(cfg, a).lower()} "
                     f"{front[a].min():.3f} to {front[a].max():.3f} {ua}, with "
                     f"{short(cfg, b).lower()} {front[b].min():.3f} to {front[b].max():.3f} {ub}.")
        ba, bb = baseline_value(data, cfg, a), baseline_value(data, cfg, b)
        if da == "max" and db == "min" and ba and bb:
            f = front[front[a] >= ba]
            if len(f):
                r = f.loc[f[b].idxmin()]
                items.append(f"At today's {short(cfg, a).lower()} ({ba:.3f} {ua}) the front needs only "
                             f"{r[b]:.2f} {ub} {short(cfg, b).lower()} (baseline {bb:.2f} {ub}, "
                             f"{100 * (r[b] / bb - 1):+.0f} %), at immersion {r.get('imm', np.nan):.2f}.")
            des = design_rows(data, cfg)
            m = des[des[a] >= 0.99 * ba]
            if len(m):
                rm = m.loc[m[b].idxmin()]
                items.append(f"Measured evidence: {m[b].idxmin()} delivers {rm[a]:.3f} {ua} with "
                             f"{rm[b]:.2f} {ub} ({100 * (rm[b] / bb - 1):+.0f} % vs baseline).")
    imp = nxt[nxt["purpose"] == "improve design"]
    exp = nxt[nxt["purpose"] == "improve model"]
    key = cfg.get("map_pair", [cfg["key_x"]])[0] if cfg.get("map_pair") else cfg["key_x"]
    if len(imp):
        items.append(f"Next runs to improve the design: {', '.join(imp.index)} at {short(cfg, key).lower()} "
                     + ", ".join(f"{v:.3g}" for v in imp[key]) + ".")
    if len(exp):
        items.append(f"Next runs to improve the model: {', '.join(exp.index)} at {short(cfg, key).lower()} "
                     + ", ".join(f"{v:.3g}" for v in exp[key]) + " (status: "
                     + ", ".join(exp["status"]) + ").")
    return md(f"**Findings for this dataset ({cfg['name']}):**\n\n" + _bullets(items))
