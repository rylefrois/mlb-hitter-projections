"""Save the analyses behind the decision memo as reports.

  hitters_to_watch_track_record.csv  How the app's ranked lists did, by
                                     comparison, stat, and forecast date,
                                     against naive lists.
  hitters_to_watch_calibration.csv   Predicted chance vs actual share, by
                                     comparison and stat.
  park_change.csv                    Whether park adjustment helps hitters
                                     whose park context changed.

Needs the app_forecasts table (forecasts.py) and model_inputs (features.py).

Usage:
    python -m mlb_ros_projections.insights
"""

import duckdb
import numpy as np
import pandas as pd

from mlb_ros_projections.backtest import KEYS, REPORTS_DIR, load_targets
from mlb_ros_projections.build_db import DB_PATH
from mlb_ros_projections.models import MODEL_SPECS, forward_chain, load_model_inputs
from mlb_ros_projections.trends import list_track_record, probability_calibration

STATS = ["woba", "k_rate", "ubb_rate"]
VERSUS = ["ytd", "marcel"]

# The app's default filter for the ranked lists.
MIN_PA_SO_FAR = 75

# Hitters whose remaining schedule's park factor differs from the parks behind
# their Marcel projection by at least this much count as changing parks.
PARK_CHANGE = 0.03


def hitters_to_watch(app: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Track record (by forecast date) and calibration (all dates) of the ranked lists."""
    app = app[app["pa_ytd"] >= MIN_PA_SO_FAR]
    months = pd.to_datetime(app["asof_date"]).dt.strftime("%b")

    records, calibrations = [], []
    for versus in VERSUS:
        for stat in STATS:
            for month in ["May", "Jun", "Jul", "Aug", "Sep"]:
                record = list_track_record(app[months == month], stat, versus=versus)
                pooled = record.groupby(["list", "method"], as_index=False).agg(
                    seasons=("season", "nunique"), hit_rate=("hit_rate", "mean"),
                    avg_actual_change=("avg_actual_change", "mean"),
                )
                records.append(pooled.assign(versus=versus, stat=stat, asof_month=month))
            calib = probability_calibration(app, stat, versus=versus)
            calibrations.append(calib.assign(versus=versus, stat=stat,
                                             bin=calib["bin"].astype(str)))
    return pd.concat(records, ignore_index=True), pd.concat(calibrations, ignore_index=True)


def park_change(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """wOBA accuracy with and without park adjustment, by how much the hitter's parks changed."""
    inputs = load_model_inputs(con)
    targets = load_targets(con).merge(
        inputs[KEYS + ["pf_ros_woba", "pf_hist_woba", "marcel_weighted_pa"]], on=KEYS
    )
    change = targets["pf_ros_woba"] / targets["pf_hist_woba"] - 1
    has_history = targets["marcel_weighted_pa"] > 0
    groups = {
        "better hitting parks (+3% or more)": has_history & (change >= PARK_CHANGE),
        "tougher parks (-3% or more)": has_history & (change <= -PARK_CHANGE),
        "similar parks (within 1%)": change.abs() < 0.01,
        "all hitters": pd.Series(True, index=targets.index),
    }

    rows = []
    for model in ["eb_playing_time", "eb_park"]:
        forecast, _ = forward_chain(inputs, MODEL_SPECS[model])
        merged = targets.merge(forecast[KEYS + ["woba"]], on=KEYS, suffixes=("", "_pred"))
        for group, mask in groups.items():
            g = merged[mask.to_numpy()]
            w = g["woba_denom_ros"]
            rows.append({
                "group": group, "model": model, "rows": len(g),
                "hitters": g[["season", "batter"]].drop_duplicates().shape[0],
                "woba_rmse": float(np.sqrt(np.average((g["woba_pred"] - g["woba"]) ** 2,
                                                      weights=w))),
                "woba_bias": float(np.average(g["woba_pred"] - g["woba"], weights=w)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    app = con.execute("SELECT * FROM app_forecasts").df()
    parks = park_change(con)
    con.close()

    record, calibration = hitters_to_watch(app)
    REPORTS_DIR.mkdir(exist_ok=True)
    record.to_csv(REPORTS_DIR / "hitters_to_watch_track_record.csv", index=False)
    calibration.to_csv(REPORTS_DIR / "hitters_to_watch_calibration.csv", index=False)
    parks.to_csv(REPORTS_DIR / "park_change.csv", index=False)

    june = record[record["asof_month"] == "Jun"]
    print("Hitters to watch, June 1 lists (share that moved as called):")
    print(june.pivot_table(index=["versus", "stat", "list"], columns="method",
                           values="hit_rate").round(3).to_string())
    print("\nPark adjustment, wOBA:")
    print(parks.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
