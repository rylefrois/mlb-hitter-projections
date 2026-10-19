import pandas as pd
import pytest

from mlb_ros_projections.calibration import bias, coverage, interval

KEY = {"season": 2024, "asof_date": pd.Timestamp("2024-06-01")}


def test_interval_is_symmetric_and_clipped_at_zero():
    lower, upper = interval(pd.Series([0.30, 0.02]), pd.Series([0.05, 0.05]), level=0.95)
    assert lower[0] == pytest.approx(0.30 - 1.96 * 0.05, abs=1e-3)
    assert upper[0] == pytest.approx(0.30 + 1.96 * 0.05, abs=1e-3)
    assert lower[1] == 0


def test_coverage_counts_inside_below_and_above():
    # Forecast 0.20 with sd 0.01 for every stat. 80% range is about 0.187 to 0.213.
    actuals = [0.20, 0.20, 0.10, 0.30]  # two inside, one below, one above
    targets = pd.DataFrame([
        {**KEY, "batter": i, "k_rate": a, "ubb_rate": a, "woba": a} for i, a in enumerate(actuals)
    ])
    forecast = pd.DataFrame([
        {**KEY, "batter": i, **{s: 0.20 for s in ["k_rate", "ubb_rate", "woba"]},
         **{f"{s}_sd": 0.01 for s in ["k_rate", "ubb_rate", "woba"]}}
        for i in range(4)
    ])
    result = coverage(targets, forecast).set_index(["stat", "level"]).loc[("k_rate", 0.8)]
    assert result["inside"] == 0.5
    assert result["below"] == 0.25
    assert result["above"] == 0.25


def test_bias_is_weighted_by_pa_left():
    # Forecast 0.20 for both. Errors +0.10 (100 PA left) and -0.10 (300 PA left).
    targets = pd.DataFrame([
        {**KEY, "batter": 0, "pa_ros": 100, "woba_denom_ros": 100,
         "k_rate": 0.10, "ubb_rate": 0.10, "woba": 0.10},
        {**KEY, "batter": 1, "pa_ros": 300, "woba_denom_ros": 300,
         "k_rate": 0.30, "ubb_rate": 0.30, "woba": 0.30},
    ])
    forecast = pd.DataFrame([
        {**KEY, "batter": i, "k_rate": 0.20, "ubb_rate": 0.20, "woba": 0.20} for i in range(2)
    ])
    # (100 * 0.10 + 300 * -0.10) / 400 = -0.05
    assert bias(targets, forecast).loc["all", "k_rate"] == pytest.approx(-0.05)
