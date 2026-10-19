"""Checks of the model zoo, the selection rule and the validation helpers.

Run with:  python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fanml as fm  # noqa: E402
import settings as S  # noqa: E402

HAS_DATA = (S.DATA_DIR / S.WORKBOOK).exists()


def toy(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = 1.0 + 0.5 * X[:, 0] - 0.8 * X[:, 0] ** 2 + 0.05 * rng.normal(size=n)
    return X, y


def test_zoo_order_and_contrast():
    zoo = fm.model_zoo(restarts=0)
    assert list(zoo) == fm.MODEL_ORDER
    assert fm.CONTRAST_ONLY == {"Random forest"}


def test_gaussian_process_returns_uncertainty_and_ignores_irrelevant_input():
    X, y = toy(30)
    gp = fm.GaussianProcessModel(restarts=2).fit(X, y)
    mu, sd = gp.predict(X[:3], return_std=True)
    assert mu.shape == sd.shape == (3,)
    assert np.all(sd > 0)
    ls = gp.length_scales()
    assert ls[1] > ls[0]          # the second input does not enter y


def test_selection_prefers_simpler_model_within_tolerance():
    chosen, best = fm._choose({"Linear": 1.04, "Quadratic": 1.0, "Gaussian process": 1.01,
                               "Random forest": 0.5}, tol=0.05)
    assert best == "Quadratic"          # random forest is never a candidate
    assert chosen == "Linear"           # within 5 % of the best and simpler
    chosen, _ = fm._choose({"Linear": 2.0, "Quadratic": 1.2, "Gaussian process": 1.0}, tol=0.05)
    assert chosen == "Gaussian process"


def test_scores_perfect_and_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert fm.scores(y, y)["q2"] == pytest.approx(1.0)
    assert fm.scores(y, np.full(4, y.mean()))["q2"] == pytest.approx(0.0)
    assert fm.scores(y, y + 0.1, scale=2.0)["rmse_pct"] == pytest.approx(5.0)


def test_loo_quadratic_recovers_toy_function():
    X, y = toy(25)
    pred, std = fm.loo_predict(fm.model_zoo()["Quadratic"], X, y, n_jobs=1)
    assert fm.scores(y, pred)["q2"] > 0.9
    assert np.isnan(std).all()


@pytest.mark.skipif(not HAS_DATA, reason="workbook not in data/raw/")
def test_later_runs_split_steel():
    cfg = fm.get_config("steel")
    data = fm.add_features(fm.load_dataset(cfg), cfg)
    res = fm.later_runs_test(data, cfg, targets=["total"], models=["Linear"])
    assert res.attrs["n_train"] == 20 and res.attrs["n_test"] == 9
    assert set(res["features"]) == {"raw", "physics"}


@pytest.mark.skipif(not HAS_DATA, reason="workbook not in data/raw/")
def test_variant_sensitivity_plastic_has_three_rows():
    cfg = fm.get_config("plastic")
    sens = fm.variant_sensitivity(cfg, "total", "Quadratic")
    assert list(sens.index) == ["recorded", "corrected", "excluded"]
    assert sens.loc["excluded", "runs"] == 21
    assert sens.loc["corrected", "q2"] > sens.loc["recorded", "q2"]   # Run 17 hurts as recorded
