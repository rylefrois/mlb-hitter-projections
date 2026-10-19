import duckdb
import pandas as pd
import pytest

from mlb_ros_projections.features import build_features

# A tiny fake season for one hitter. Each tuple is one pitch:
# (game_date, game_pk, at_bat_number, events, woba_value, woba_denom, xwoba)
PITCHES = [
    ("2024-04-10", 1, 1, None, None, None, None),            # not the last pitch of the PA
    ("2024-04-10", 1, 1, "strikeout", 0.0, 1, 0.0),
    ("2024-04-10", 1, 2, "walk", 0.7, 1, 0.7),
    ("2024-04-11", 2, 1, "home_run", 2.0, 1, 1.6),
    ("2024-04-11", 2, 2, "intent_walk", 0.4, None, None),   # PA, but not in wOBA
    ("2024-04-11", 2, 3, "truncated_pa", 0.0, 0, None),      # not a real PA
    ("2024-06-15", 3, 1, "single", 0.9, 1, None),            # missing xwOBA
]


@pytest.fixture
def con():
    df = pd.DataFrame(
        PITCHES,
        columns=["game_date", "game_pk", "at_bat_number", "events",
                 "woba_value", "woba_denom", "estimated_woba_using_speedangle"],
    )
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["batter"] = 100
    df["pitcher"] = 200
    df["stand"] = "R"
    df["p_throws"] = "L"
    df["home_team"] = "AAA"
    df["away_team"] = "BBB"
    df["inning_topbot"] = "Top"

    con = duckdb.connect()
    con.register("pitches_df", df)
    con.execute("CREATE TABLE pitches AS SELECT * FROM pitches_df")
    build_features(con)
    return con


def snapshot(con, asof):
    return con.execute(
        "SELECT * FROM batter_snapshot WHERE asof_date = ?", [asof]
    ).df().iloc[0]


def test_pa_table_keeps_only_completed_plate_appearances(con):
    events = {r[0] for r in con.execute("SELECT events FROM pa").fetchall()}
    assert events == {"strikeout", "walk", "home_run", "intent_walk", "single"}


def test_batting_team_comes_from_inning_half(con):
    assert con.execute("SELECT DISTINCT bat_team FROM pa").fetchall() == [("BBB",)]


def test_snapshot_splits_on_asof_date(con):
    s = snapshot(con, "2024-05-01")
    assert s.pa_ytd == 4
    assert s.k_ytd == 1
    assert s.ubb_ytd == 1                       # intentional walk excluded
    assert s.woba_denom_ytd == 3                # intentional walk excluded
    assert s.woba_num_ytd == pytest.approx(2.7)
    assert s.xwoba_num_ytd == pytest.approx(2.3)
    assert s.pa_ros == 1
    assert s.xwoba_num_ros == pytest.approx(0.9)  # fell back to actual wOBA


def test_snapshot_after_last_game_has_no_future(con):
    s = snapshot(con, "2024-07-01")
    assert s.pa_ytd == 5
    assert s.pa_ros == 0


def test_one_snapshot_per_month_may_through_september(con):
    assert con.execute("SELECT count(*) FROM snapshot_dates").fetchone()[0] == 5
