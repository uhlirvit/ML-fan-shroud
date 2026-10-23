"""Checks of the interpretation, prediction and design helpers (sections 7-9).

Run with:  python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fanml as fm  # noqa: E402
import settings as S  # noqa: E402

HAS_DATA = (S.DATA_DIR / S.WORKBOOK).exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="workbook not in data/raw/")


def test_inverse_maps_round_trip():
    for name in ("steel", "plastic"):
        cfg = fm.get_config(name)
        rng = np.random.default_rng(0)
        raw = pd.DataFrame({c: rng.uniform(*cfg["bounds"][c], 50) for c in cfg["inputs"]})
        phys = fm.physics_from_raw(raw, cfg)
        back = fm.raw_from_physics(phys[cfg["physics_features"]], cfg)
        for c in cfg["inputs"]:
            assert np.allclose(back[c], raw[c]), (name, c)


def test_conformal_quantile_and_kinds():
    idx = [f"Run {i}" for i in range(19)]
    err = np.arange(1, 20, dtype=float)            # absolute errors 1..19
    table = pd.DataFrame({"measured": np.zeros(19), "predicted": err, "std": np.nan, "error": err}, index=idx)
    gp = table.assign(std=2.0)
    cv = fm.CVResult(pd.DataFrame(), {("a", "physics", "Quadratic"): table, ("b", "physics", "Gaussian process"): gp})
    sel = pd.DataFrame({"model": ["Quadratic", "Gaussian process"], "features": ["physics", "physics"]},
                       index=pd.Index(["a", "b"], name="target"))
    conf = fm.conformal_intervals(cv, sel, level=0.9)
    # k = ceil(20 * 0.9) = 18 -> 18th smallest error
    assert conf.loc["a", "kind"] == "absolute" and conf.loc["a", "q"] == 18.0
    assert conf.loc["b", "kind"] == "normalised" and conf.loc["b", "q"] == 9.0
    assert not conf["capped"].any()


def test_expected_improvement_properties():
    mu = np.array([1.0, 1.0, 2.0])
    sd = np.array([0.1, 1.0, 0.1])
    ei = fm.expected_improvement(mu, sd, best=1.5)
    assert np.all(ei >= 0)
    assert ei[1] > ei[0]          # more uncertainty, more chance to beat the best
    assert ei[2] == pytest.approx(0.5, abs=1e-3)


@needs_data
def test_range_status_categories_steel():
    cfg = fm.get_config("steel")
    data = fm.add_features(fm.load_dataset(cfg), cfg)
    chk = fm.range_checker(data, cfg)
    designs = fm.physics_from_raw(pd.DataFrame([
        {"ring": 0.0, "trans": 0.0, "tri_large": 0.5, "tri_small": 0.5},     # baseline: inside
        {"ring": 40.0, "trans": 40.0, "tri_large": 0.5, "tri_small": 0.5},   # immersion 1.52
        {"ring": 90.0, "trans": 0.0, "tri_large": 0.5, "tri_small": 0.5},    # ring above its limit
    ]), cfg)
    st = fm.range_status(designs, cfg, chk)["status"].tolist()
    assert st == ["inside tested range", "outside tested range", "outside design limits"]


@needs_data
def test_design_pool_respects_limits_and_range():
    cfg = fm.get_config("plastic")
    data = fm.add_features(fm.load_dataset(cfg), cfg)
    pool = fm.design_pool(data, cfg, n=1024)
    assert len(pool) > 50
    assert (pool["status"] == "inside tested range").all()
    for c in cfg["inputs"]:
        lo, hi = cfg["bounds"][c]
        assert pool[c].between(lo, hi).all()


@needs_data
def test_next_runs_ldr700():
    cfg = fm.get_config("ldr700")
    data = fm.add_features(fm.load_dataset(cfg), cfg)
    cv = fm.cross_validate(data, cfg, models=["Quadratic", "Gaussian process"])
    sel = fm.select_models(cv, cfg)
    models = fm.fit_final_models(data, cfg, sel)
    conf = fm.conformal_intervals(cv, sel)
    nxt = fm.suggest_next_runs(data, cfg, models, conf, n_improve=2, n_explore=1)
    assert list(nxt["purpose"]) == ["improve design", "improve design", "improve model"]
    lo, hi = cfg["bounds"]["p1"]
    assert nxt["p1"].between(lo, hi).all()
    assert abs(nxt["p1"].iloc[0] - nxt["p1"].iloc[1]) > 0.05     # the batch spreads out
