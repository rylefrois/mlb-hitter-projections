"""Create the DuckDB database with views over the raw Parquet files.

Usage:
    python -m mlb_ros_projections.build_db
"""

from pathlib import Path

import duckdb

DB_PATH = Path("data/mlb.duckdb")


def build() -> None:
    con = duckdb.connect(str(DB_PATH))
    # A view reads the Parquet files on demand: no copy of the data, and
    # newly downloaded seasons appear automatically.
    con.execute("""
        CREATE OR REPLACE VIEW pitches AS
        SELECT * FROM read_parquet('data/raw/statcast_*.parquet')
    """)
    con.execute("""
        CREATE OR REPLACE VIEW people AS
        SELECT * FROM read_parquet('data/raw/people.parquet')
    """)
    con.execute("""
        CREATE OR REPLACE VIEW venues AS
        SELECT * FROM read_parquet('data/raw/venues.parquet')
    """)
    con.close()
    print(f"built {DB_PATH}")


if __name__ == "__main__":
    build()
