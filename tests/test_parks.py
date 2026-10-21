import duckdb
import pandas as pd
import pytest

from mlb_ros_projections.features import run_sql_file


def games(season, game_pk, home, away, pa, k, date):
    """`pa` plate appearances in one game, the first `k` of them strikeouts."""
    return [
        {"season": season, "game_date": pd.Timestamp(date).date(), "game_pk": game_pk,
         "at_bat_number": i, "batter": 1 if i % 2 else 2, "home_team": home, "away_team": away,
         "bat_team": home if i % 2 else away, "is_k": i < k, "is_ubb": False,
         "woba_denom": 1, "woba_value": 0.5 if i < k else 0.25}
        for i in range(pa)
    ]


@pytest.fixture
def con():
    # 2018: games at AAA strike out 30%, AAA's road games (at BBB) 20%.
    # 2019: one game at each park, so 2019 is a target season with 2018 history.
    rows = (
        games(2018, 1, "AAA", "BBB", 100, 30, "2018-06-01")
        + games(2018, 2, "BBB", "AAA", 100, 20, "2018-06-02")
        + games(2019, 3, "AAA", "BBB", 10, 3, "2019-04-15")
        + games(2019, 4, "BBB", "AAA", 10, 2, "2019-06-15")
    )
    con = duckdb.connect()
    con.register("pa_df", pd.DataFrame(rows))
    con.execute("CREATE TABLE pa AS SELECT * FROM pa_df")
    con.execute("CREATE TABLE snapshot_dates AS SELECT 2019 AS season, DATE '2019-05-01' AS asof_date")
    run_sql_file(con, "parks.sql")
    return con


def factor(con, park):
    return con.execute(
        "SELECT k_rate FROM park_factors WHERE season = 2019 AND park = ?", [park]
    ).fetchone()[0]


def test_park_factor_compares_home_games_with_same_teams_road_games(con):
    assert factor(con, "AAA") == pytest.approx(0.30 / 0.20)
    assert factor(con, "BBB") == pytest.approx(0.20 / 0.30)


def test_factors_use_only_earlier_seasons(con):
    # 2018 has no earlier seasons, so it gets no factors at all.
    assert con.execute("SELECT count(*) FROM park_factors WHERE season = 2018").fetchone()[0] == 0


def test_remaining_schedule_uses_games_after_the_forecast_date(con):
    # Before May 1 2019 both hitters played only at AAA; afterward only at BBB.
    ytd, ros = con.execute("""
        SELECT y.pf_ytd_k_rate, r.pf_ros_k_rate
        FROM park_exposure_ytd AS y JOIN park_exposure_ros AS r USING (season, asof_date, batter)
        WHERE batter = 1
    """).fetchone()
    assert ytd == pytest.approx(1.5)
    assert ros == pytest.approx(0.20 / 0.30)
