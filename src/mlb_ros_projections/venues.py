"""Download the ballpark (venue) for every regular-season game from the MLB Stats API.

Statcast identifies games by home team, not ballpark, so a team that moves
(the Athletics to Sacramento and the Rays to Steinbrenner Field in 2025) or
plays at a neutral site (Tokyo, London, Bristol) would get the wrong park.
Park factors use the real venue instead.

Usage:
    python -m mlb_ros_projections.venues --start 2015 --end 2026
"""

import argparse

import pandas as pd
import requests

from mlb_ros_projections.ingest import RAW_DIR

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
VENUES_PATH = RAW_DIR / "venues.parquet"


def fetch_season_venues(year: int) -> pd.DataFrame:
    response = requests.get(
        SCHEDULE_URL, params={"sportId": 1, "season": year, "gameType": "R"}, timeout=60
    )
    response.raise_for_status()
    rows = [
        {
            "game_pk": game["gamePk"],
            "season": year,
            "venue_id": game["venue"]["id"],
            "venue_name": game["venue"]["name"],
        }
        for day in response.json()["dates"]
        for game in day["games"]
        if game["status"]["abstractGameState"] == "Final"
    ]
    # Suspended games appear once per day they were played; keep where they finished.
    return pd.DataFrame(rows).drop_duplicates("game_pk", keep="last")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    venues = pd.concat(
        [fetch_season_venues(year) for year in range(args.start, args.end + 1)],
        ignore_index=True,
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    venues.to_parquet(VENUES_PATH, index=False)
    print(f"saved {len(venues):,} games at {venues['venue_id'].nunique()} venues to {VENUES_PATH}")


if __name__ == "__main__":
    main()
