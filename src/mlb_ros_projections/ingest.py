"""Download Statcast pitch-level data and save one Parquet file per season.

Usage:
    python -m mlb_ros_projections.ingest --start 2023 --end 2024
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pybaseball

RAW_DIR = Path("data/raw")

# The regular season always falls between March and October. We fetch
# month by month because long date ranges are slow and tend to time out.
SEASON_MONTHS = range(3, 11)

# Columns later steps depend on. We save everything, but fail loudly if
# Baseball Savant ever drops or renames one of these.
REQUIRED_COLUMNS = [
    "game_date", "game_pk", "game_type", "at_bat_number", "pitch_number",
    "batter", "pitcher", "events", "woba_value", "woba_denom",
    "estimated_woba_using_speedangle", "home_team", "stand", "p_throws",
]


def month_ranges(year: int) -> list[tuple[str, str]]:
    """Return (start, end) date strings covering each season month."""
    ranges = []
    for month in SEASON_MONTHS:
        start = date(year, month, 1)
        end = date(year, month + 1, 1) - timedelta(days=1)
        ranges.append((start.isoformat(), end.isoformat()))
    return ranges


def fetch_range(start: str, end: str, attempts: int = 4) -> pd.DataFrame | None:
    """Fetch one date range, retrying when Savant returns an error page instead of CSV."""
    for attempt in range(1, attempts + 1):
        try:
            return pybaseball.statcast(start_dt=start, end_dt=end, verbose=False)
        except Exception as exc:
            if attempt == attempts:
                raise
            wait = 30 * attempt
            print(f"  {start} to {end} failed ({type(exc).__name__}); retrying in {wait}s")
            time.sleep(wait)


def fetch_season(year: int) -> pd.DataFrame:
    """Fetch every regular-season pitch for one year."""
    chunks = []
    for start, end in month_ranges(year):
        print(f"  fetching {start} to {end}")
        df = fetch_range(start, end)
        if df is not None and not df.empty:
            chunks.append(df)

    season = pd.concat(chunks, ignore_index=True)

    # Date ranges pick up spring training ("S") and postseason games; keep "R" only.
    season = season[season["game_type"] == "R"]

    # Month boundaries can occasionally return the same pitch twice.
    season = season.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])

    missing = set(REQUIRED_COLUMNS) - set(season.columns)
    if missing:
        raise ValueError(f"{year}: Statcast is missing expected columns {sorted(missing)}")

    # pybaseball returns game_date as text; store a real date so SQL date functions work.
    season["game_date"] = pd.to_datetime(season["game_date"]).dt.date

    return season.reset_index(drop=True)


def save_season(df: pd.DataFrame, year: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"statcast_{year}.parquet"
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="re-download saved seasons")
    args = parser.parse_args()

    pybaseball.cache.enable()

    for year in range(args.start, args.end + 1):
        path = RAW_DIR / f"statcast_{year}.parquet"
        if path.exists() and not args.force:
            print(f"{year}: already saved, skipping")
            continue
        print(f"{year}: downloading")
        df = fetch_season(year)
        save_season(df, year)
        print(f"{year}: saved {len(df):,} pitches to {path}")


if __name__ == "__main__":
    main()
