"""Check whether forecast ranges are honest: do 80% ranges contain the actual
result about 80% of the time?

Usage:
    python -m mlb_ros_projections.calibration
"""

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

from mlb_ros_projections.backtest import KEYS, REPORTS_DIR, STATS, load_targets
from mlb_ros_projections.build_db import DB_PATH
from mlb_ros_projections.models import (
    FINAL_MODEL,
    MODEL_SPECS,
    forward_chain,
    load_model_inputs,
)

LEVELS = [0.5, 0.8, 0.95]


def interval(forecast: pd.Series, sd: pd.Series, level: float) -> tuple[pd.Series, pd.Series]:
    """Central range holding `level` of the forecast distribution (rates can't go below 0)."""
    z = norm.ppf(0.5 + level / 2)
    return (forecast - z * sd).clip(lower=0), forecast + z * sd


def coverage(
    targets: pd.DataFrame, forecast: pd.DataFrame, sd_suffix: str = "_sd", by: str | None = None
) -> pd.DataFrame:
    """For each stat and level: share of actuals inside, below, and above the range.

    A well-calibrated 80% range has ~80% inside and ~10% on each side.
    """
    merged = targets.merge(forecast, on=KEYS, suffixes=("", "_pred"))
    group = by or "overall"
    merged = merged.assign(overall="all")

    rows = []
    for stat in STATS:
        pred, sd, actual = merged[f"{stat}_pred"], merged[f"{stat}{sd_suffix}"], merged[stat]
        for level in LEVELS:
            lower, upper = interval(pred, sd, level)
            flags = pd.DataFrame({
                group: merged[group],
                "inside": (actual >= lower) & (actual <= upper),
                "below": actual < lower,
                "above": actual > upper,
                "width": upper - lower,
            })
            for value, g in flags.groupby(group, observed=True):
                rows.append({
                    "stat": stat, group: value, "level": level, "rows": len(g),
                    "inside": g["inside"].mean(), "below": g["below"].mean(),
                    "above": g["above"].mean(), "avg_width": g["width"].mean(),
                })
    return pd.DataFrame(rows)


def bias(targets: pd.DataFrame, forecast: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Average of forecast minus actual, weighted by PA left. Positive = forecasts too high.

    Ranges can be well calibrated overall while still being centered in the
    wrong place for some group of hitters, so this is checked separately.
    """
    merged = targets.merge(forecast, on=KEYS, suffixes=("", "_pred")).assign(overall="all")
    group = by or "overall"
    rows = []
    for value, g in merged.groupby(group, observed=True):
        row = {group: value, "rows": len(g)}
        for stat, weight_col in STATS.items():
            row[stat] = np.average(g[f"{stat}_pred"] - g[stat], weights=g[weight_col])
        rows.append(row)
    return pd.DataFrame(rows).set_index(group)


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    targets = load_targets(con)
    inputs = load_model_inputs(con)
    con.close()

    forecast, params = forward_chain(inputs, MODEL_SPECS[FINAL_MODEL])

    print(f"Model: {FINAL_MODEL}. Share of actual results inside each range "
          f"(target = the level):\n")
    raw = coverage(targets, forecast, sd_suffix="_sd_raw")
    corrected = coverage(targets, forecast)
    summary = raw[["stat", "level", "inside"]].rename(columns={"inside": "before_correction"})
    summary["after_correction"] = corrected["inside"].to_numpy()
    summary["below"] = corrected["below"].to_numpy()
    summary["above"] = corrected["above"].to_numpy()
    summary["avg_width"] = corrected["avg_width"].to_numpy()
    print(summary.round(3).to_string(index=False))

    print("\nWidth correction learned from earlier seasons (1 = no change, >1 = widen):")
    scales = params.pivot(index="season", columns="stat", values="interval_scale")
    print(scales[list(STATS)].round(3).to_string())

    REPORTS_DIR.mkdir(exist_ok=True)
    corrected.to_csv(REPORTS_DIR / "calibration_overall.csv", index=False)
    for by in ["asof_month", "pa_ytd_bucket"]:
        table = coverage(targets, forecast, by=by)
        table.to_csv(REPORTS_DIR / f"calibration_{by}.csv", index=False)
        order = targets[by].cat.categories
        eighty = table[table["level"] == 0.8].pivot(index=by, columns="stat", values="inside")
        print(f"\n80% range coverage by {by}:")
        print(eighty.reindex(order)[list(STATS)].round(3).to_string())

    print("\nBias: forecast minus actual, weighted by PA left (positive = too high):")
    overall_bias = bias(targets, forecast)
    print(overall_bias.round(4).to_string())
    by_bucket = bias(targets, forecast, by="pa_ytd_bucket")
    print(by_bucket.round(4).to_string())
    pd.concat([overall_bias, by_bucket]).to_csv(REPORTS_DIR / "calibration_bias.csv")


if __name__ == "__main__":
    main()
