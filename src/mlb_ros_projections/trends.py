"""Which hitters' current numbers misstate their real level?

"Slumps to ignore" are hitters likely to do better than their season-to-date
numbers from here on; "hot starts to doubt" are hitters likely to do worse.
Each hitter's chance of improving comes from the forecast range (the same
ranges the calibration check validates), and the lists can be backtested
against what actually happened in past seasons.

All functions take rows shaped like the `app_forecasts` table.
"""

import pandas as pd
from scipy.stats import norm

from mlb_ros_projections.models import sd_from_parts

# +1 when a higher rate is better for the hitter, -1 when lower is better.
DIRECTION = {"woba": 1, "k_rate": -1, "ubb_rate": 1}

# Gaps smaller than these aren't worth mentioning as a reason.
NOTABLE_GAP = {"woba": 0.015, "k_rate": 0.02, "ubb_rate": 0.01}

# A forecast needs this many PA left to be judged against what happened.
MIN_PA_LEFT = 50

# The sample-size column each stat's rest-of-season rate is measured over.
PA_LEFT_COLUMN = {"woba": "woba_denom_ros", "k_rate": "pa_ros", "ubb_rate": "pa_ros"}


def improvement_probability(
    df: pd.DataFrame, stat: str, remaining_pa: float | pd.Series
) -> pd.Series:
    """Chance the hitter's next `remaining_pa` PA beat their season-to-date rate."""
    sd = sd_from_parts(
        df[f"{stat}_per_pa_var"], df[f"{stat}_evidence"], remaining_pa,
        df[f"{stat}_interval_scale"],
    )
    gain = DIRECTION[stat] * (df[stat] - df[f"ytd_{stat}"])
    return pd.Series(norm.cdf(gain / sd), index=df.index)


def expected_change(df: pd.DataFrame, stat: str) -> pd.Series:
    """Forecast minus season to date, signed so positive means the hitter does better."""
    return DIRECTION[stat] * (df[stat] - df[f"ytd_{stat}"])


def reasons(row: pd.Series, stat: str) -> str:
    """Plain-language drivers of the gap between the forecast and season to date."""
    direction, threshold = DIRECTION[stat], NOTABLE_GAP[stat]
    out = []

    track_record = direction * (row[f"marcel_{stat}"] - row[f"ytd_{stat}"])
    if abs(track_record) >= threshold:
        out.append("track record is better" if track_record > 0 else "track record is worse")

    if stat == "woba" and pd.notna(row["ytd_xwoba"]):
        contact_luck = row["ytd_xwoba"] - row["ytd_woba"]
        if abs(contact_luck) >= threshold:
            out.append("unlucky on contact (xwOBA higher)" if contact_luck > 0
                       else "lucky on contact (xwOBA lower)")

    # A park factor above 1 raises the stat, which helps for wOBA/BB and hurts for K.
    park_change = direction * (row[f"pf_ros_{stat}"] / row[f"pf_ytd_{stat}"] - 1)
    if abs(park_change) >= 0.02:
        out.append("easier parks ahead" if park_change > 0 else "tougher parks ahead")

    if not out:
        out.append("small sample: expect a pull toward the average")
    return "; ".join(out)


def rank_candidates(
    df: pd.DataFrame, stat: str, remaining_pa: float, top_n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(most likely to improve, most likely to decline), each with probability and reasons."""
    ranked = df.assign(
        p_improve=improvement_probability(df, stat, remaining_pa),
        expected_change=expected_change(df, stat),
    )
    improvers = ranked.nlargest(top_n, "p_improve")
    decliners = ranked.nsmallest(top_n, "p_improve")
    for table in (improvers, decliners):
        table["reasons"] = [reasons(row, stat) for _, row in table.iterrows()]
    return improvers, decliners


def scoreable(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Rows with enough PA left to judge the forecast against what happened."""
    return df[(df["pa_ros"] >= MIN_PA_LEFT) & df[f"actual_{stat}"].notna()]


def list_track_record(df: pd.DataFrame, stat: str, top_n: int = 10) -> pd.DataFrame:
    """How past lists did, season by season, against a naive list.

    For each season, the model's top-N "improve" and "decline" lists are
    compared with naive lists: the N worst (or best) hitters so far. Bad
    starts improve partly by regression to the mean alone, so the model has
    to beat the naive list to show it adds anything.

    Probabilities here use each hitter's actual PA left, since that's what
    the outcome is measured over.
    """
    df = scoreable(df, stat)
    df = df.assign(
        actual_change=DIRECTION[stat] * (df[f"actual_{stat}"] - df[f"ytd_{stat}"]),
        p_improve=improvement_probability(df, stat, df[PA_LEFT_COLUMN[stat]]),
    )
    worst_so_far = DIRECTION[stat] * df[f"ytd_{stat}"]  # low = worst for the hitter

    rows = []
    for season, g in df.groupby("season"):
        picks = {
            ("improve", "model"): g.nlargest(top_n, "p_improve"),
            ("improve", "naive"): g.loc[worst_so_far.loc[g.index].nsmallest(top_n).index],
            ("decline", "model"): g.nsmallest(top_n, "p_improve"),
            ("decline", "naive"): g.loc[worst_so_far.loc[g.index].nlargest(top_n).index],
        }
        for (direction, method), picked in picks.items():
            moved_as_called = picked["actual_change"] > 0 if direction == "improve" \
                else picked["actual_change"] < 0
            rows.append({
                "season": season, "list": direction, "method": method,
                "hit_rate": moved_as_called.mean(),
                "avg_actual_change": picked["actual_change"].mean(),
            })
    return pd.DataFrame(rows)


def probability_calibration(
    df: pd.DataFrame, stat: str, bins: tuple[float, ...] = (0, 0.2, 0.4, 0.6, 0.8, 1)
) -> pd.DataFrame:
    """Predicted chance of improving vs the share that actually improved."""
    df = scoreable(df, stat)
    p = improvement_probability(df, stat, df[PA_LEFT_COLUMN[stat]])
    improved = DIRECTION[stat] * (df[f"actual_{stat}"] - df[f"ytd_{stat}"]) > 0
    table = pd.DataFrame({"p": p, "improved": improved,
                          "bin": pd.cut(p, bins, include_lowest=True)})
    return (table.groupby("bin", observed=True)
            .agg(hitters=("p", "size"), predicted=("p", "mean"), actual=("improved", "mean"))
            .reset_index())
