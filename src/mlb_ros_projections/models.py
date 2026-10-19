"""Empirical-Bayes rest-of-season forecasts: Marcel updated with this season's data.

For each stat, the forecast blends this season's numbers with Marcel as if
Marcel were worth `strength` plate appearances of evidence:

    forecast = (events_ytd + strength * prior) / (pa_ytd + strength)

This is the posterior mean of a beta-binomial model. The parameters (prior
strength, and optionally the xwOBA and environment weights) are fit by
minimizing the same weighted squared error the backtest scores.

Parameters are fit by forward-chaining: each season is forecast with
parameters learned only from earlier seasons, so no future data is used.
"""

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mlb_ros_projections.backtest import EVAL_SEASONS, KEYS, STATS, is_scored

# The first season with a Marcel projection is 2016 (from 2015 history).
FIRST_TRAIN_SEASON = 2016
EXCLUDED_SEASONS = {2020}

# Marcel's own regression constant. A hitter's Marcel reliability is
# weighted_pa / (weighted_pa + 1200): 0 for a rookie, about 0.85 for a regular.
MARCEL_REGRESSION_PA = 1200

# Events column and sample-size column for each stat's season-to-date numbers.
YTD_COLUMNS = {
    "k_rate": ("k_ytd", "pa_ytd"),
    "ubb_rate": ("ubb_ytd", "pa_ytd"),
    "woba": ("woba_num_ytd", "woba_denom_ytd"),
}


@dataclass(frozen=True)
class ModelSpec:
    """Which ideas a model version includes, so each can be scored on its own."""

    history_prior: bool = False   # prior strength depends on how much history backs Marcel
    use_xwoba: bool = False       # wOBA forecasts also use this season's xwOBA
    environment: bool = False     # shift the prior when the league environment changes


MODEL_SPECS = {
    "eb_basic": ModelSpec(),
    "eb_history": ModelSpec(history_prior=True),
    "eb_full": ModelSpec(history_prior=True, use_xwoba=True, environment=True),
}


def load_model_inputs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("SELECT * FROM model_inputs").df()


def param_bounds(stat: str, spec: ModelSpec) -> dict[str, tuple[float, float]]:
    """The parameters a model version fits for one stat, with their allowed ranges."""
    bounds = {}
    if spec.history_prior:
        bounds["strength_no_history"] = (1.0, 3000.0)
        bounds["strength_full_history"] = (1.0, 3000.0)
    else:
        bounds["strength"] = (1.0, 3000.0)
    if spec.use_xwoba and stat == "woba":
        bounds["xwoba_weight"] = (0.0, 1.0)
    if spec.environment:
        bounds["environment_weight"] = (0.0, 1.5)
    return bounds


def predict(df: pd.DataFrame, stat: str, params: dict[str, float], spec: ModelSpec) -> np.ndarray:
    events_col, trials_col = YTD_COLUMNS[stat]
    trials = df[trials_col].to_numpy(dtype=float)
    events = df[events_col].to_numpy(dtype=float)

    if spec.use_xwoba and stat == "woba":
        # xwOBA reflects contact quality, which is less noisy than results.
        w = params["xwoba_weight"]
        events = (1 - w) * events + w * df["xwoba_num_ytd"].to_numpy(dtype=float)

    prior = df[f"marcel_{stat}"].to_numpy(dtype=float)
    if spec.environment:
        factor = df[f"{stat}_factor"].to_numpy(dtype=float)
        prior = prior * (1 + params["environment_weight"] * (factor - 1))

    if spec.history_prior:
        wpa = df["marcel_weighted_pa"].to_numpy(dtype=float)
        reliability = wpa / (wpa + MARCEL_REGRESSION_PA)
        lo, hi = params["strength_no_history"], params["strength_full_history"]
        strength = lo + (hi - lo) * reliability
    else:
        strength = params["strength"]

    return (events + strength * prior) / (trials + strength)


def fit(train: pd.DataFrame, stat: str, spec: ModelSpec) -> dict[str, float]:
    """Find the parameters that minimize weighted squared error on the training rows."""
    bounds = param_bounds(stat, spec)
    names = list(bounds)
    actual = train[f"actual_{stat}"].to_numpy(dtype=float)
    weight = train[STATS[stat]].to_numpy(dtype=float)

    # Prior strengths are searched on a log scale, so 50 -> 100 is as easy a
    # step for the optimizer as 500 -> 1000.
    is_log = np.array([name.startswith("strength") for name in names])

    def to_params(x):
        return dict(zip(names, np.where(is_log, np.exp(x), x)))

    def mse(x):
        pred = predict(train, stat, to_params(x), spec)
        return np.average((pred - actual) ** 2, weights=weight)

    search_bounds = [(np.log(lo), np.log(hi)) if log else (lo, hi)
                     for log, (lo, hi) in zip(is_log, bounds.values())]
    start = np.array([np.log(200.0) if log else (lo + hi) / 2
                      for log, (lo, hi) in zip(is_log, bounds.values())])

    # Squared errors on rates are tiny (~0.002), and L-BFGS-B's stopping rules
    # are on an absolute scale, so it would quit at the start. Scale the loss
    # so the starting point is 1.
    scale = mse(start)
    result = minimize(lambda x: mse(x) / scale, start, bounds=search_bounds, method="L-BFGS-B")
    return to_params(result.x)


def forward_chain(
    inputs: pd.DataFrame, spec: ModelSpec, seasons: list[int] = EVAL_SEASONS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast each season with parameters fit only on earlier seasons.

    Returns (forecasts, fitted parameters by season and stat).
    """
    trainable = (
        inputs["season"].ge(FIRST_TRAIN_SEASON)
        & ~inputs["season"].isin(EXCLUDED_SEASONS)
        & is_scored(inputs)
    )
    forecasts, fitted = [], []
    for season in seasons:
        train = inputs[trainable & (inputs["season"] < season)]
        test = inputs[inputs["season"] == season]
        forecast = test[KEYS].copy()
        for stat in STATS:
            params = fit(train, stat, spec)
            forecast[stat] = predict(test, stat, params, spec)
            fitted.append({"season": season, "stat": stat, "train_rows": len(train), **params})
        forecasts.append(forecast)
    return pd.concat(forecasts, ignore_index=True), pd.DataFrame(fitted)


def model_forecasts(
    con: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Forecasts and fitted parameters for every model version in MODEL_SPECS."""
    inputs = load_model_inputs(con)
    forecasts, params = {}, {}
    for name, spec in MODEL_SPECS.items():
        forecasts[name], params[name] = forward_chain(inputs, spec)
    return forecasts, params
