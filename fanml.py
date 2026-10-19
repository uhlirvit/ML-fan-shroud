"""fanml - analysis helpers for the fan-shroud ML framework.

The notebook ``fan_ml_framework.ipynb`` calls these functions; every dataset-specific
number lives in ``settings.py``.

Sections
    1. Settings access, output folders and plotting style
    2. Load and validate                      (notebook section 1)
    3. Features                               (notebook section 4)
    4. Exploratory analysis                   (notebook section 2)
    5. Structure (unsupervised)               (notebook section 3)
    6. Findings text for the takeaway cells
"""
from __future__ import annotations

import platform
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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
