import numpy as np
import pandas as pd
import pytest

from mlb_ros_projections.models import ModelSpec, fit, predict

BASIC = ModelSpec()


def one_row(**overrides):
    row = {
        "pa_ytd": 100, "k_ytd": 30, "ubb_ytd": 8,
        "woba_denom_ytd": 90, "woba_num_ytd": 30.0, "xwoba_num_ytd": 27.0,
        "marcel_k_rate": 0.20, "marcel_ubb_rate": 0.08, "marcel_woba": 0.320,
        "marcel_weighted_pa": 0.0,
        "k_rate_factor": 1.0, "ubb_rate_factor": 1.0, "woba_factor": 1.0,
    }
    return pd.DataFrame([{**row, **overrides}])


def test_forecast_is_weighted_average_of_season_and_prior():
    # (30 + 100 * 0.20) / (100 + 100) = 0.25: halfway between 0.30 and 0.20.
    pred = predict(one_row(), "k_rate", {"strength": 100}, BASIC)
    assert pred[0] == pytest.approx(0.25)


def test_no_data_this_season_returns_the_prior():
    pred = predict(one_row(pa_ytd=0, k_ytd=0), "k_rate", {"strength": 100}, BASIC)
    assert pred[0] == pytest.approx(0.20)


def test_history_prior_trusts_marcel_more_for_veterans():
    spec = ModelSpec(history_prior=True)
    params = {"strength_no_history": 50, "strength_full_history": 450}
    rookie = predict(one_row(marcel_weighted_pa=0), "k_rate", params, spec)[0]
    veteran = predict(one_row(marcel_weighted_pa=6000), "k_rate", params, spec)[0]
    # Same 30% start: the veteran's forecast stays closer to the 20% prior.
    assert abs(veteran - 0.20) < abs(rookie - 0.20)


def test_xwoba_weight_blends_actual_and_expected():
    spec = ModelSpec(use_xwoba=True)
    full_xwoba = predict(one_row(), "woba", {"strength": 1e-9, "xwoba_weight": 1.0}, spec)
    assert full_xwoba[0] == pytest.approx(27.0 / 90)


def test_environment_shifts_the_prior():
    spec = ModelSpec(environment=True)
    params = {"strength": 1e9, "environment_weight": 1.0}  # forecast is all prior
    pred = predict(one_row(k_rate_factor=1.1), "k_rate", params, spec)
    assert pred[0] == pytest.approx(0.22)


def test_fit_recovers_known_prior_strength():
    # Simulate hitters whose true K rate varies around their prior with SD tau.
    # Bayes says the best prior strength is p(1-p) / tau^2.
    rng = np.random.default_rng(0)
    n, prior, tau = 40_000, 0.22, 0.04
    talent = np.clip(rng.normal(prior, tau, n), 0.01, 0.99)
    pa_ytd, pa_ros = 100, 400
    train = pd.DataFrame({
        "pa_ytd": pa_ytd,
        "k_ytd": rng.binomial(pa_ytd, talent),
        "pa_ros": pa_ros,
        "actual_k_rate": rng.binomial(pa_ros, talent) / pa_ros,
        "marcel_k_rate": prior,
    })
    expected = prior * (1 - prior) / tau**2  # about 107 PA

    params = fit(train, "k_rate", BASIC)

    assert params["strength"] == pytest.approx(expected, rel=0.15)
