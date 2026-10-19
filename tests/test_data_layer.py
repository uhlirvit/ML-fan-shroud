"""Checks of the data layer and the exploration helpers.

Run with:  python -m pytest -q
The tests need the workbook in data/raw/; they are skipped when it is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fanml as fm  # noqa: E402
import settings as S  # noqa: E402

pytestmark = pytest.mark.skipif(not (S.DATA_DIR / S.WORKBOOK).exists(),
                                reason="workbook not in data/raw/")


def load(name, variant="recorded"):
    cfg = fm.get_config(name)
    return cfg, fm.load_dataset(cfg, variant=variant)


@pytest.mark.parametrize("name, rows", [("steel", 30), ("plastic", 22), ("ldr700", 7)])
def test_every_sheet_loads(name, rows):
    cfg, df = load(name)
    assert len(df) == rows
    assert set(cfg["outputs"]) <= set(df.columns)
    assert df[cfg["outputs"]].notna().all().all()


def test_columns_are_mapped_by_name_not_position():
    # The plastic sheet orders P1, P3, P2: Run 3 has Ring (P3) = 44.32 and Transition (P2) = -17.08.
    _, df = load("plastic")
    assert df.loc["Run 3", "ring"] == pytest.approx(44.32)
    assert df.loc["Run 3", "trans"] == pytest.approx(-17.08)


def test_run17_total_is_flagged():
    cfg, df = load("plastic")
    report = fm.validate(df, cfg)
    assert report.flagged_runs == ["Run 17"]
    row = report.checks.set_index("check").loc["Components add up to the total"]
    assert row["status"] == "FLAG"


@pytest.mark.parametrize("name", ["steel", "plastic"])
def test_immersion_formula_matches_sheet(name):
    cfg, df = load(name)
    report = fm.validate(df, cfg)
    assert report.details["formula_imm"].max() < 1e-9


def test_steel_baseline_has_no_inputs_and_is_kept_out_of_design_rows():
    cfg, df = load("steel")
    assert df.loc["Baseline", "is_baseline"]
    assert "Baseline" not in fm.design_rows(df, cfg).index
    b = fm.baseline_point(df, cfg)
    assert b["imm"] == pytest.approx(77.8 / 103.8)


def test_variants_for_run17():
    cfg, corrected = load("plastic", "corrected")
    comps = corrected.loc["Run 17", cfg["components"]].sum()
    assert corrected.loc["Run 17", "total"] == pytest.approx(comps)
    _, excluded = load("plastic", "excluded")
    assert "Run 17" not in excluded.index
    with pytest.raises(ValueError):
        fm.load_dataset(cfg, variant="no-such-variant")


@pytest.mark.parametrize("name", ["steel", "plastic"])
def test_fractions_sum_to_one(name):
    cfg, df = load(name)
    data = fm.add_features(df, cfg)
    fracs = data[[f"frac_{c}" for c in cfg["components"]]].sum(axis=1)
    assert np.allclose(fracs, 1.0)


def test_steel_largest_immersion_gap():
    cfg, df = load("steel")
    data = fm.add_features(df, cfg)
    g = fm.find_gaps(fm.design_rows(data, cfg)["imm"]).iloc[0]
    assert g["start"] == pytest.approx(0.3719, abs=1e-3)
    assert g["end"] == pytest.approx(0.5628, abs=1e-3)


def test_immersion_is_exact_combination_of_raw_depths():
    cfg, df = load("steel")
    check = fm.collinearity_check(fm.add_features(df, cfg), cfg)
    assert check.attrs["imm_r2_from_raw"] == pytest.approx(1.0)
    assert check.loc["raw inputs + immersion", "condition number"] > 1e10
    assert check.loc["physics features", "condition number"] < 10


def test_compress_runs():
    assert fm.compress_runs(["Run 1", "Run 2", "Run 3", "Run 17"]) == "Runs 1–3, 17"
    assert fm.compress_runs(["Run 17"]) == "Run 17"
    assert fm.compress_runs(["Baseline"]) == "Baseline"


def test_sig_digits():
    assert fm.sig_digits(1.5761) == 5
    assert fm.sig_digits(0.30554990780358) == 14
    assert fm.sig_digits(2.0) == 1
