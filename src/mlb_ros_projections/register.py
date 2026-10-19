"""Download the Chadwick Bureau player register (names, IDs, birthdates).

pybaseball's chadwick_register() drops the birthdate columns, which Marcel
needs for its age adjustment, so we read the register files directly.

Usage:
    python -m mlb_ros_projections.register
"""

import io
import zipfile

import pandas as pd
import requests

from mlb_ros_projections.ingest import RAW_DIR

REGISTER_URL = "https://github.com/chadwickbureau/register/archive/refs/heads/master.zip"
PEOPLE_PATH = RAW_DIR / "people.parquet"

COLUMNS = [
    "key_mlbam", "key_fangraphs", "key_bbref", "name_first", "name_last",
    "birth_year", "birth_month", "birth_day", "mlb_played_first", "mlb_played_last",
]


def fetch_people() -> pd.DataFrame:
    """Return one row per person with an MLBAM id (the id Statcast uses)."""
    response = requests.get(REGISTER_URL, timeout=120)
    response.raise_for_status()

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    people_files = [f for f in archive.namelist() if "/data/people-" in f and f.endswith(".csv")]
    people = pd.concat(
        (pd.read_csv(archive.open(f), usecols=COLUMNS, low_memory=False) for f in people_files),
        ignore_index=True,
    )

    people = people.dropna(subset=["key_mlbam"])
    int_columns = ["key_mlbam", "key_fangraphs", "birth_year", "birth_month", "birth_day",
                   "mlb_played_first", "mlb_played_last"]
    people[int_columns] = people[int_columns].astype("Int64")
    return people.reset_index(drop=True)


def main() -> None:
    people = fetch_people()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    people.to_parquet(PEOPLE_PATH, index=False)
    print(f"saved {len(people):,} people to {PEOPLE_PATH}")


if __name__ == "__main__":
    main()
