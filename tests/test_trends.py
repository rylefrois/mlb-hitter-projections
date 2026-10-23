import pandas as pd
import pytest

from mlb_ros_projections.trends import (
    improvement_probability,
    list_track_record,
    probability_calibration,
    rank_candidates,
    reasons,
)


def hitter(**overrides):
    """One hitter-snapshot shaped like an app_forecasts row, neutral by default."""
    row = {
        "season": 2024, "batter": 1, "name": "Test Hitter", "pa_ros": 300, "woba_denom_ros": 270,
        "ytd_xwoba": 0.320,
        "pf_ytd_woba": 1.0, "pf_ros_woba": 1.0, "pf_ytd_k_rate": 1.0, "pf_ros_k_rate": 1.0,
        "pf_ytd_ubb_rate": 1.0, "pf_ros_ubb_rate": 1.0,
    }
    for stat, value in {"woba": 0.320, "k_rate": 0.22, "ubb_rate": 0.08}.items():
        row |= {
            stat: value, f"ytd_{stat}": value, f"marcel_{stat}": value, f"actual_{stat}": value,
            f"{stat}_per_pa_var": 0.25 if stat == "woba" else value * (1 - value),
            f"{stat}_evidence": 600.0, f"{stat}_interval_scale": 1.0,
        }
    return row | overrides


def test_forecast_equal_to_season_to_date_is_a_coin_flip():
    df = pd.DataFrame([hitter()])
    assert improvement_probability(df, "woba", 150)[0] == pytest.approx(0.5)


def test_more_pa_ahead_makes_the_call_more_certain():
    df = pd.DataFrame([hitter(woba=0.340, ytd_woba=0.300)])
    assert improvement_probability(df, "woba", 600)[0] > improvement_probability(df, "woba", 50)[0]


def test_strikeouts_improve_by_going_down():
    df = pd.DataFrame([hitter(k_rate=0.20, ytd_k_rate=0.28)])
    assert improvement_probability(df, "k_rate", 150)[0] > 0.5


def test_reasons_name_track_record_contact_luck_and_parks():
    row = pd.Series(hitter(ytd_woba=0.280, marcel_woba=0.340, ytd_xwoba=0.330,
                           pf_ytd_woba=0.95, pf_ros_woba=1.02))
    text = reasons(row, "woba")
    assert "track record is better" in text
    assert "unlucky on contact" in text
    assert "easier parks ahead" in text


def test_reasons_fall_back_to_regression_when_nothing_stands_out():
    assert "small sample" in reasons(pd.Series(hitter()), "woba")


def test_rank_candidates_splits_improvers_and_decliners():
    df = pd.DataFrame([
        hitter(batter=1, woba=0.340, ytd_woba=0.280),   # forecast well above: improve
        hitter(batter=2, woba=0.300, ytd_woba=0.380),   # forecast well below: decline
        hitter(batter=3),
    ])
    improvers, decliners = rank_candidates(df, "woba", remaining_pa=150, top_n=1)
    assert improvers["batter"].tolist() == [1]
    assert decliners["batter"].tolist() == [2]


def test_track_record_scores_model_and_naive_lists():
    # Hitter 1: model expects improvement and it happens. Hitter 2 has the worst
    # start (the naive pick) but the model expects it to continue, and it does.
    df = pd.DataFrame([
        hitter(batter=1, ytd_woba=0.290, woba=0.340, actual_woba=0.350),
        hitter(batter=2, ytd_woba=0.250, woba=0.250, actual_woba=0.240),
        hitter(batter=3, ytd_woba=0.330, woba=0.330, actual_woba=0.330),
    ])
    result = list_track_record(df, "woba", top_n=1).set_index(["list", "method"])
    assert result.loc[("improve", "model"), "hit_rate"] == 1.0
    assert result.loc[("improve", "naive"), "hit_rate"] == 0.0


def test_hitters_with_too_few_pa_left_are_not_scored():
    df = pd.DataFrame([hitter(pa_ros=20, woba_denom_ros=18)])
    assert probability_calibration(df, "woba").empty
