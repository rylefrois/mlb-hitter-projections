"""Build feature, baseline, and model-input tables in DuckDB from the raw views.

Usage:
    python -m mlb_ros_projections.features
"""

from pathlib import Path

import duckdb

from mlb_ros_projections.build_db import DB_PATH

SQL_DIR = Path(__file__).parent

# Run in this order: each file reads tables the earlier ones create.
SQL_FILES = ["features.sql", "baselines.sql", "parks.sql", "model_inputs.sql"]

TABLES = [
    "pa", "batter_day", "batter_season", "snapshot_dates", "batter_snapshot",
    "league_season", "batter_age", "marcel", "batter_baselines",
    "game_venue", "park_factors", "park_exposure_ytd", "park_exposure_ros", "park_exposure_hist",
    "league_snapshot", "environment", "model_inputs",
]


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str) -> None:
    con.execute((SQL_DIR / name).read_text())


def build_features(con: duckdb.DuckDBPyConnection) -> None:
    """Run features.sql against a connection that has a `pitches` table or view."""
    run_sql_file(con, "features.sql")


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    for name in SQL_FILES:
        run_sql_file(con, name)
    for table in TABLES:
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n:,} rows")
    con.close()


if __name__ == "__main__":
    main()
