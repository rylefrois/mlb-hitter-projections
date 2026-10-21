import numpy as np
import pandas as pd
import pytest

from mlb_ros_projections.models import (
    ModelSpec,
    fit,
    fit_interval_scale,
    predict,
    predictive_sd,
)

BASIC = ModelSpec()


def one_row(**overrides):
    row = {
        "pa_ytd": 100, "k_ytd": 30, "ubb_ytd": 8,
        "woba_denom_ytd": 90, "woba_num_ytd": 30.0, "xwoba_num_ytd": 27.0,
        "marcel_k_rate": 0.20, "marcel_ubb_rate": 0.08, "marcel_woba": 0.320,
        "marcel_weighted_pa": 0.0,
        "k_rate_factor": 1.0, "ubb_rate_factor": 1.0, "woba_factor": 1.0,
        "pa_ytd_share": 1.0,
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


def test_playing_time_shifts_prior_for_fringe_hitters():
    spec = ModelSpec(playing_time_prior=True)
    params = {"strength": 1e9, "shift_no_history": 0.10, "shift_part_time": 0.20}  # all prior
    # Regular with full history: wpa so large reliability is ~1, full-time share.
    regular = predict(one_row(marcel_weighted_pa=1e9), "k_rate", params, spec)[0]
    # Rookie playing half the time: 0.20 * (1 + 0.10 * 1 + 0.20 * 0.5) = 0.24.
    rookie = predict(one_row(marcel_weighted_pa=0, pa_ytd_share=0.5), "k_rate", params, spec)[0]
    assert regular == pytest.approx(0.20)
    assert rookie == pytest.approx(0.24)


def test_playing_time_share_above_one_counts_as_full_time():
    spec = ModelSpec(playing_time_prior=True)
    params = {"strength": 1e9, "shift_no_history": 0.0, "shift_part_time": 0.20}
    pred = predict(one_row(pa_ytd_share=1.3), "k_rate", params, spec)[0]
    assert pred == pytest.approx(0.20)


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


def test_predictive_sd_adds_talent_and_luck_variance():
    # Forecast 0.25: per-PA variance 0.1875. Evidence 100 PA + 100 prior = 200; 300 PA left.
    df = one_row(pa_ros=300, woba_denom_ros=270)
    sd = predict_sd_for(df, "k_rate", 0.25)
    assert sd == pytest.approx(np.sqrt(0.1875 / 200 + 0.1875 / 300))


def test_predictive_sd_for_woba_uses_previous_season_per_pa_variance():
    df = one_row(pa_ros=300, woba_denom_ros=270, prev_woba_pa_var=0.25)
    sd = predict_sd_for(df, "woba", 0.330)
    assert sd == pytest.approx(np.sqrt(0.25 / (90 + 100) + 0.25 / 270))


def test_interval_scale_is_one_when_errors_match_the_model():
    # Simulate actuals from exactly the distribution the model describes.
    rng = np.random.default_rng(1)
    n = 50_000
    df = pd.concat([one_row(pa_ros=300, woba_denom_ros=270)] * n, ignore_index=True)
    params = {"strength": 100}
    forecast = predict(df, "k_rate", params, BASIC)
    sd = predictive_sd(df, "k_rate", forecast, params, BASIC)
    df["actual_k_rate"] = forecast + rng.normal(0, sd)
    assert fit_interval_scale(df, "k_rate", params, BASIC) == pytest.approx(1.0, abs=0.02)


def predict_sd_for(df, stat, forecast):
    return predictive_sd(df, stat, np.array([forecast]), {"strength": 100}, BASIC)[0]
