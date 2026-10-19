import numpy as np
import pandas as pd
import pytest

from mlb_ros_projections.backtest import evaluate, weighted_rmse

KEY = {"season": 2024, "asof_date": pd.Timestamp("2024-06-01")}


def make_targets():
    # Two hitters: one with 400 PA left, one with 100.
    return pd.DataFrame([
        {**KEY, "batter": 1, "pa_ros": 400, "woba_denom_ros": 360,
         "k_rate": 0.20, "ubb_rate": 0.08, "woba": 0.330, "asof_month": "Jun"},
        {**KEY, "batter": 2, "pa_ros": 100, "woba_denom_ros": 90,
         "k_rate": 0.30, "ubb_rate": 0.06, "woba": 0.290, "asof_month": "Jun"},
    ])


def make_forecast(k1, k2):
    return pd.DataFrame([
        {**KEY, "batter": 1, "k_rate": k1, "ubb_rate": 0.08, "woba": 0.330},
        {**KEY, "batter": 2, "k_rate": k2, "ubb_rate": 0.06, "woba": 0.290},
    ])


NOISE = pd.DataFrame([{"season": 2024, "k_rate": 0.16, "ubb_rate": 0.07, "woba": 0.25}])


def test_weighted_rmse_weights_by_sample_size():
    # Errors of 0.1 (weight 3) and 0.2 (weight 1): sqrt((3*0.01 + 0.04) / 4)
    got = weighted_rmse(pd.Series([0.1, 0.2]), pd.Series([0.0, 0.0]), pd.Series([3, 1]))
    assert got == pytest.approx(np.sqrt(0.07 / 4))


def test_perfect_forecast_has_zero_error_and_beats_marcel():
    forecasts = {"marcel": make_forecast(0.25, 0.25), "perfect": make_forecast(0.20, 0.30)}
    results = evaluate(make_targets(), forecasts, NOISE).set_index(["forecast", "stat"])
    assert results.loc[("perfect", "k_rate"), "rmse"] == 0
    assert results.loc[("perfect", "k_rate"), "skill_vs_marcel"] == pytest.approx(1.0)
    assert results.loc[("marcel", "k_rate"), "skill_vs_marcel"] == pytest.approx(0.0)


def test_noise_floor_is_per_pa_variance_over_sample_size():
    # Weighted mean of 0.16/n with weights n is 0.16 * rows / total PA.
    results = evaluate(make_targets(), {"marcel": make_forecast(0.2, 0.3)}, NOISE)
    k = results.set_index("stat").loc["k_rate"]
    assert k["noise_floor"] == pytest.approx(np.sqrt(0.16 * 2 / 500))


def test_results_can_be_split_by_a_column():
    results = evaluate(make_targets(), {"marcel": make_forecast(0.2, 0.3)}, NOISE, by="asof_month")
    assert set(results["asof_month"]) == {"Jun"}


def test_forecast_missing_rows_is_rejected():
    partial = make_forecast(0.2, 0.3).iloc[:1]
    with pytest.raises(ValueError, match="no k_rate forecast for 1 target rows"):
        evaluate(make_targets(), {"marcel": partial}, NOISE)
