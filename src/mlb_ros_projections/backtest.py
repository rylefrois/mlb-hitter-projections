"""Score rest-of-season forecasts against what actually happened.

Every forecast is scored on the same rows: hitters with at least one PA
before the forecast date and MIN_PA_ROS after it, for every forecast date
in EVAL_SEASONS. A forecast must cover every row, so a model can't look
better by skipping the hard cases.

A forecast is a DataFrame with key columns (season, asof_date, batter) and
one column per stat in STATS holding the predicted rest-of-season rate.

Usage:
    python -m mlb_ros_projections.backtest
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from mlb_ros_projections.build_db import DB_PATH

KEYS = ["season", "asof_date", "batter"]

# 2015-2017 are history for Marcel. 2020 is excluded: the season started in
# late July, so the May-July forecast dates have no data, and 60 games is
# too few for a fair test.
EVAL_SEASONS = [2018, 2019, 2021, 2022, 2023, 2024]

# Below this, the "actual" rest-of-season rate is mostly noise.
MIN_PA_ROS = 50

# Each stat and the column that measures its sample size. Errors are
# weighted by that sample size, so a 400-PA miss counts more than a 60-PA one.
STATS = {"k_rate": "pa_ros", "ubb_rate": "pa_ros", "woba": "woba_denom_ros"}

PA_YTD_BUCKETS = [1, 50, 150, 300, np.inf]
PA_YTD_LABELS = ["1-49", "50-149", "150-299", "300+"]

REPORTS_DIR = Path("reports")


def is_scored(df: pd.DataFrame) -> pd.Series:
    """Which hitter-snapshots count: forecasts are scored on these, and models train on them."""
    return (
        (df["pa_ytd"] >= 1)
        & (df["woba_denom_ytd"] >= 1)  # else season-to-date wOBA is undefined (e.g. only IBBs)
        & (df["pa_ros"] >= MIN_PA_ROS)
    )


def load_targets(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The rows every forecast is scored on, with rest-of-season actuals."""
    targets = con.execute(
        """
        SELECT season, asof_date, batter, pa_ytd, woba_denom_ytd, pa_ros, woba_denom_ros,
               actual_k_rate AS k_rate, actual_ubb_rate AS ubb_rate, actual_woba AS woba
        FROM batter_baselines
        WHERE season IN (SELECT unnest(?))
        """,
        [EVAL_SEASONS],
    ).df()
    targets = targets[is_scored(targets)].reset_index(drop=True)
    targets["asof_month"] = pd.Categorical(
        pd.to_datetime(targets["asof_date"]).dt.strftime("%b"),
        categories=["May", "Jun", "Jul", "Aug", "Sep"],
        ordered=True,
    )
    targets["pa_ytd_bucket"] = pd.cut(
        targets["pa_ytd"], PA_YTD_BUCKETS, labels=PA_YTD_LABELS, right=False
    )
    return targets


def load_baseline_forecasts(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """The three baselines from baselines.sql, in the standard forecast format."""
    season_to_date = con.execute("""
        SELECT season, asof_date, batter,
               ytd_k_rate AS k_rate, ytd_ubb_rate AS ubb_rate, ytd_woba AS woba
        FROM batter_baselines
    """).df()
    marcel = con.execute("""
        SELECT season, asof_date, batter,
               marcel_k_rate AS k_rate, marcel_ubb_rate AS ubb_rate, marcel_woba AS woba
        FROM batter_baselines
    """).df()
    # Same number for everyone: the prior seasons' league rates Marcel regresses toward.
    league_average = con.execute("""
        SELECT b.season, b.asof_date, b.batter, l.k_rate, l.ubb_rate, l.woba
        FROM batter_baselines AS b
        JOIN marcel_league AS l USING (season)
    """).df()
    return {
        "league_average": league_average,
        "season_to_date": season_to_date,
        "marcel": marcel,
    }


def noise_variance(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-PA outcome variance by season, for each stat.

    Even a forecaster who knew every hitter's true talent would miss by this
    much, because outcomes over a few hundred PA are random. It sets the
    noise floor: the best RMSE any model could hope for.
    """
    return con.execute("""
        SELECT season,
               var_pop(is_k::INT)   AS k_rate,
               var_pop(is_ubb::INT) AS ubb_rate,
               var_pop(woba_value) FILTER (WHERE woba_denom = 1) AS woba
        FROM pa
        GROUP BY season
    """).df()


def weighted_rmse(pred: pd.Series, actual: pd.Series, weight: pd.Series) -> float:
    return float(np.sqrt(np.average((pred - actual) ** 2, weights=weight)))


def evaluate(
    targets: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    noise: pd.DataFrame,
    by: str | None = None,
) -> pd.DataFrame:
    """Weighted RMSE for each forecast and stat, optionally split by a targets column.

    skill_vs_marcel is 1 - MSE / MSE_marcel: 0 means no better than Marcel,
    positive means better, negative means worse.
    """
    group = by or "overall"
    targets = targets.assign(overall="all")
    season_noise = noise.set_index("season")

    rows = []
    for name, forecast in forecasts.items():
        merged = targets.merge(forecast, on=KEYS, how="left", suffixes=("", "_pred"))
        for stat, weight_col in STATS.items():
            missing = merged[f"{stat}_pred"].isna().sum()
            if missing:
                raise ValueError(f"{name} has no {stat} forecast for {missing} target rows")

            # Variance of a rate over n PA is (per-PA variance) / n.
            merged["_noise"] = merged["season"].map(season_noise[stat]) / merged[weight_col]

            for value, g in merged.groupby(group, observed=True):
                w = g[weight_col]
                rows.append({
                    "forecast": name,
                    "stat": stat,
                    group: value,
                    "rows": len(g),
                    "rmse": weighted_rmse(g[f"{stat}_pred"], g[stat], w),
                    "noise_floor": float(np.sqrt(np.average(g["_noise"], weights=w))),
                })

    results = pd.DataFrame(rows)
    marcel = (
        results[results["forecast"] == "marcel"]
        .set_index(["stat", group])["rmse"]
        .rename("marcel_rmse")
    )
    results = results.join(marcel, on=["stat", group])
    results["skill_vs_marcel"] = 1 - results["rmse"] ** 2 / results["marcel_rmse"] ** 2
    return results.drop(columns="marcel_rmse")


def print_table(results: pd.DataFrame, by: str | None = None) -> None:
    group = by or "overall"
    for stat in STATS:
        table = results[results["stat"] == stat]
        wide = table.pivot(index=group, columns="forecast", values="rmse")
        wide = wide[table["forecast"].unique()]  # keep the order forecasts were given in
        wide["noise_floor"] = table.groupby(group, observed=True)["noise_floor"].first()
        wide = wide.reindex(table[group].unique())
        print(f"\n{stat} RMSE" + (f" by {by}" if by else ""))
        print(wide.round(4).to_string())


def main() -> None:
    # Imported here because models.py imports this module's scoring rules.
    from mlb_ros_projections.models import model_forecasts

    con = duckdb.connect(str(DB_PATH), read_only=True)
    targets = load_targets(con)
    forecasts = load_baseline_forecasts(con)
    models, params = model_forecasts(con)
    forecasts.update(models)
    noise = noise_variance(con)
    con.close()

    print(f"{len(targets):,} hitter-snapshots across seasons {EVAL_SEASONS}")

    REPORTS_DIR.mkdir(exist_ok=True)
    for by in [None, "asof_month", "pa_ytd_bucket"]:
        results = evaluate(targets, forecasts, noise, by=by)
        print_table(results, by=by)
        results.to_csv(REPORTS_DIR / f"backtest_{by or 'overall'}.csv", index=False)

    all_params = pd.concat(
        [p.assign(model=name) for name, p in params.items()], ignore_index=True
    )
    all_params.to_csv(REPORTS_DIR / "model_params.csv", index=False)
    print("\nFitted parameters (eb_playing_time), by forecast season:")
    final = all_params[all_params["model"] == "eb_playing_time"].drop(columns="model")
    print(final.dropna(axis=1, how="all").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
