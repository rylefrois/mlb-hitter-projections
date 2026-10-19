"""Build hitter feature tables in DuckDB from the raw pitches view.

Usage:
    python -m mlb_ros_projections.features
"""

from pathlib import Path

import duckdb

from mlb_ros_projections.build_db import DB_PATH

SQL_PATH = Path(__file__).with_name("features.sql")


def build_features(con: duckdb.DuckDBPyConnection) -> None:
    """Run features.sql against a connection that has a `pitches` table or view."""
    con.execute(SQL_PATH.read_text())


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    build_features(con)
    for table in ["pa", "batter_day", "batter_season", "snapshot_dates", "batter_snapshot"]:
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n:,} rows")
    con.close()


if __name__ == "__main__":
    main()
