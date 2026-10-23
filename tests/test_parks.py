import duckdb
import pandas as pd
import pytest

from mlb_ros_projections.features import run_sql_file

# Venue ids for the fake ballparks.
AAA_PARK, BBB_PARK, NEW_PARK = 1, 2, 3


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
    # 2018: games at AAA's park strike out 30%; AAA's road games (at BBB) 20%.
    # 2019: AAA plays one home game at its old park, then moves to a new park.
    rows = (
        games(2018, 1, "AAA", "BBB", 3000, 900, "2018-06-01")
        + games(2018, 2, "BBB", "AAA", 3000, 600, "2018-06-02")
        + games(2019, 3, "AAA", "BBB", 10, 3, "2019-04-15")
        + games(2019, 4, "BBB", "AAA", 10, 2, "2019-06-15")
        + games(2019, 5, "AAA", "BBB", 10, 2, "2019-06-20")
    )
    venues = pd.DataFrame([
        {"game_pk": 1, "season": 2018, "venue_id": AAA_PARK, "venue_name": "Old Park"},
        {"game_pk": 2, "season": 2018, "venue_id": BBB_PARK, "venue_name": "BBB Park"},
        {"game_pk": 3, "season": 2019, "venue_id": AAA_PARK, "venue_name": "Old Park"},
        {"game_pk": 4, "season": 2019, "venue_id": BBB_PARK, "venue_name": "BBB Park"},
        {"game_pk": 5, "season": 2019, "venue_id": NEW_PARK, "venue_name": "New Park"},
    ])
    con = duckdb.connect()
    con.register("pa_df", pd.DataFrame(rows))
    con.register("venues_df", venues)
    con.execute("CREATE TABLE pa AS SELECT * FROM pa_df")
    con.execute("CREATE TABLE venues AS SELECT * FROM venues_df")
    con.execute("CREATE TABLE snapshot_dates AS SELECT 2019 AS season, DATE '2019-05-01' AS asof_date")
    run_sql_file(con, "parks.sql")
    return con


def factor(con, venue_id):
    row = con.execute(
        "SELECT k_rate FROM park_factors WHERE season = 2019 AND venue_id = ?", [venue_id]
    ).fetchone()
    return row[0] if row else None


def test_park_factor_compares_home_games_with_host_teams_road_games(con):
    assert factor(con, AAA_PARK) == pytest.approx(0.30 / 0.20)
    assert factor(con, BBB_PARK) == pytest.approx(0.20 / 0.30)


def test_factors_use_only_earlier_seasons(con):
    # 2018 has no earlier seasons, so it gets no factors at all.
    assert con.execute("SELECT count(*) FROM park_factors WHERE season = 2018").fetchone()[0] == 0


def test_new_ballpark_has_no_factor_even_for_a_team_with_history(con):
    # AAA moved: its old park's factor must not carry over to the new one.
    assert factor(con, NEW_PARK) is None


def test_remaining_schedule_uses_games_after_the_forecast_date(con):
    # Before May 1 2019 both hitters played only at AAA's old park (factor 1.5).
    # Afterward: one game at BBB (0.667) and one at the new park (neutral, 1.0).
    ytd, ros = con.execute("""
        SELECT y.pf_ytd_k_rate, r.pf_ros_k_rate
        FROM park_exposure_ytd AS y JOIN park_exposure_ros AS r USING (season, asof_date, batter)
        WHERE batter = 1
    """).fetchone()
    assert ytd == pytest.approx(1.5)
    assert ros == pytest.approx((0.20 / 0.30 + 1.0) / 2)
