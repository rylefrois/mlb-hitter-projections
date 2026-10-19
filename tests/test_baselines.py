import duckdb
import pandas as pd
import pytest

from mlb_ros_projections.features import run_sql_file

# Hitter A strikes out 20% of the time, B 30%, every season 2016-2018.
# C debuts in 2019 with no history. Every season has 1000 league PA at 25%,
# so the league rate Marcel regresses toward is 25%.
SEASONS = [
    # (season, batter, pa, k)
    *[(y, "A", 500, 100) for y in (2016, 2017, 2018)],
    *[(y, "B", 500, 150) for y in (2016, 2017, 2018)],
    (2019, "A", 500, 100),
    (2019, "B", 500, 150),
    (2019, "C", 100, 25),
]

PEOPLE = [
    # (key_mlbam, birth_year, birth_month, birth_day)
    ("A", 1990, 3, 1),    # age 29 in 2019: no age adjustment
    ("B", 1995, 8, 15),   # born after June 30, so age 23 in 2019, not 24
]


@pytest.fixture
def con():
    seasons = pd.DataFrame(SEASONS, columns=["season", "batter", "pa", "k"])
    seasons["ubb"] = seasons["pa"] // 10
    seasons["woba_denom"] = seasons["pa"]
    seasons["woba_num"] = seasons["pa"] * 0.32
    seasons["xwoba_num"] = seasons["pa"] * 0.32
    people = pd.DataFrame(PEOPLE, columns=["key_mlbam", "birth_year", "birth_month", "birth_day"])

    con = duckdb.connect()
    con.register("seasons_df", seasons)
    con.register("people_df", people)
    con.execute("CREATE TABLE batter_season AS SELECT * FROM seasons_df")
    con.execute("CREATE TABLE people AS SELECT * FROM people_df")
    # batter_baselines also reads snapshots; an empty table is enough here.
    con.execute("""
        CREATE TABLE batter_snapshot (
            season INT, asof_date DATE, batter VARCHAR, pa_ytd INT, pa_ros INT,
            k_ytd INT, ubb_ytd INT, woba_denom_ytd INT, woba_num_ytd DOUBLE, xwoba_num_ytd DOUBLE,
            k_ros INT, ubb_ros INT, woba_denom_ros INT, woba_num_ros DOUBLE
        )
    """)
    run_sql_file(con, "baselines.sql")
    return con


def marcel_row(con, batter):
    return con.execute(
        "SELECT * FROM marcel WHERE season = 2019 AND batter = ?", [batter]
    ).df().iloc[0]


def test_age_uses_june_30_cutoff(con):
    assert marcel_row(con, "A").age == 29
    assert marcel_row(con, "B").age == 23


def test_marcel_regresses_toward_league(con):
    # Weighted PA = (5+4+3) * 500 = 6000; weighted K = 1200.
    # (1200 + 1200 * 0.25) / (6000 + 1200) = 0.208333...
    a = marcel_row(con, "A")
    assert a.marcel_weighted_pa == 6000
    assert a.marcel_k_rate == pytest.approx(1500 / 7200)


def test_marcel_age_adjustment_lowers_strikeouts_for_young_hitters(con):
    # Raw: (1800 + 300) / 7200. Age 23 is 6 years under 29: factor 1 + 0.006 * 6.
    b = marcel_row(con, "B")
    assert b.marcel_k_rate == pytest.approx((2100 / 7200) / 1.036)


def test_hitter_with_no_history_gets_league_average(con):
    c = marcel_row(con, "C")
    assert c.marcel_weighted_pa == 0
    assert c.marcel_k_rate == pytest.approx(0.25)


def test_first_season_has_no_marcel(con):
    # Nothing before 2016 to project from.
    assert con.execute("SELECT count(*) FROM marcel WHERE season = 2016").fetchone()[0] == 0
