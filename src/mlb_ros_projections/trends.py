"""Which hitters will do better or worse than a reference number from here on?

Two comparisons, chosen with `versus`:

  "ytd"     Against the hitter's season-to-date numbers: whose current line
            misstates their level? "Slumps to ignore" are likely to do better
            from here; "hot starts to doubt" are likely to do worse.
  "marcel"  Against the preseason (Marcel) projection: whose outlook has
            changed? "Outlooks raised" are likely to beat their preseason
            projection; "outlooks lowered" are likely to fall short.

Each hitter's chance comes from the forecast range (the same ranges the
calibration check validates), and the lists can be backtested against what
actually happened in past seasons.

All functions take rows shaped like the `app_forecasts` table.
"""

import pandas as pd
from scipy.stats import norm

from mlb_ros_projections.models import sd_from_parts

# +1 when a higher rate is better for the hitter, -1 when lower is better.
DIRECTION = {"woba": 1, "k_rate": -1, "ubb_rate": 1}

# What the forecast is compared with: column prefix for each `versus` option.
REFERENCE = {"ytd": "ytd", "marcel": "marcel"}

# Gaps smaller than these aren't worth mentioning as a reason.
NOTABLE_GAP = {"woba": 0.015, "k_rate": 0.02, "ubb_rate": 0.01}

# A forecast needs this many PA left to be judged against what happened.
MIN_PA_LEFT = 50

# The sample-size column each stat's rest-of-season rate is measured over.
PA_LEFT_COLUMN = {"woba": "woba_denom_ros", "k_rate": "pa_ros", "ubb_rate": "pa_ros"}


def reference(df: pd.DataFrame, stat: str, versus: str) -> pd.Series:
    return df[f"{REFERENCE[versus]}_{stat}"]


def improvement_probability(
    df: pd.DataFrame, stat: str, remaining_pa: float | pd.Series, versus: str = "ytd"
) -> pd.Series:
    """Chance the hitter's next `remaining_pa` PA beat the reference number."""
    sd = sd_from_parts(
        df[f"{stat}_per_pa_var"], df[f"{stat}_evidence"], remaining_pa,
        df[f"{stat}_interval_scale"],
    )
    gain = DIRECTION[stat] * (df[stat] - reference(df, stat, versus))
    return pd.Series(norm.cdf(gain / sd), index=df.index)


def expected_change(df: pd.DataFrame, stat: str, versus: str = "ytd") -> pd.Series:
    """Forecast minus the reference, signed so positive means the hitter does better."""
    return DIRECTION[stat] * (df[stat] - reference(df, stat, versus))


def _park_reason(row: pd.Series, stat: str, before: str) -> str | None:
    # A park factor above 1 raises the stat, which helps for wOBA/BB and hurts for K.
    change = DIRECTION[stat] * (row[f"pf_ros_{stat}"] / row[f"pf_{before}_{stat}"] - 1)
    if abs(change) < 0.02:
        return None
    return "easier parks ahead" if change > 0 else "tougher parks ahead"


def reasons(row: pd.Series, stat: str, versus: str = "ytd") -> str:
    """Plain-language drivers of the gap between the forecast and the reference."""
    direction, threshold = DIRECTION[stat], NOTABLE_GAP[stat]
    out = []

    if versus == "ytd":
        track_record = direction * (row[f"marcel_{stat}"] - row[f"ytd_{stat}"])
        if abs(track_record) >= threshold:
            out.append("track record is better" if track_record > 0 else "track record is worse")
        if stat == "woba" and pd.notna(row["ytd_xwoba"]):
            contact_luck = row["ytd_xwoba"] - row["ytd_woba"]
            if abs(contact_luck) >= threshold:
                out.append("unlucky on contact (xwOBA higher)" if contact_luck > 0
                           else "lucky on contact (xwOBA lower)")
        park = _park_reason(row, stat, before="ytd")
        fallback = "small sample: expect a pull toward the average"
    else:
        start = direction * (row[f"ytd_{stat}"] - row[f"marcel_{stat}"])
        if abs(start) >= threshold:
            out.append("strong start" if start > 0 else "weak start")
        if stat == "woba" and pd.notna(row["ytd_xwoba"]):
            contact = row["ytd_xwoba"] - row["marcel_woba"]
            if abs(contact) >= threshold:
                out.append("contact quality above track record" if contact > 0
                           else "contact quality below track record")
        if row["pa_ytd_share"] < 0.5:
            out.append("part-time role")
        park = _park_reason(row, stat, before="hist")
        fallback = "close to preseason projection"

    if park:
        out.append(park)
    return "; ".join(out) if out else fallback


def rank_candidates(
    df: pd.DataFrame, stat: str, remaining_pa: float, top_n: int = 10, versus: str = "ytd"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(most likely to beat the reference, most likely to fall short), with reasons."""
    ranked = df.assign(
        p_improve=improvement_probability(df, stat, remaining_pa, versus),
        expected_change=expected_change(df, stat, versus),
    )
    improvers = ranked.nlargest(top_n, "p_improve")
    decliners = ranked.nsmallest(top_n, "p_improve")
    for table in (improvers, decliners):
        table["reasons"] = [reasons(row, stat, versus) for _, row in table.iterrows()]
    return improvers, decliners


def scoreable(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Rows with enough PA left to judge the forecast against what happened."""
    return df[(df["pa_ros"] >= MIN_PA_LEFT) & df[f"actual_{stat}"].notna()]


def naive_score(df: pd.DataFrame, stat: str, versus: str) -> pd.Series:
    """What a naive list ranks by: higher = more likely to do better, naively.

    ytd:    the worst hitters so far will bounce back (regression to the mean).
    marcel: hitters running furthest above their projection will keep it up
            (taking the start at face value).
    """
    if versus == "ytd":
        return -DIRECTION[stat] * df[f"ytd_{stat}"]
    return DIRECTION[stat] * (df[f"ytd_{stat}"] - df[f"marcel_{stat}"])


def list_track_record(
    df: pd.DataFrame, stat: str, top_n: int = 10, versus: str = "ytd"
) -> pd.DataFrame:
    """How past lists did, season by season, against a naive list.

    For each season, the model's top-N lists in each direction are compared
    with naive lists (see `naive_score`). The model has to beat the naive
    list to show it adds anything.

    Probabilities here use each hitter's actual PA left, since that's what
    the outcome is measured over.
    """
    df = scoreable(df, stat)
    df = df.assign(
        actual_change=DIRECTION[stat] * (df[f"actual_{stat}"] - reference(df, stat, versus)),
        p_improve=improvement_probability(df, stat, df[PA_LEFT_COLUMN[stat]], versus),
        naive=naive_score(df, stat, versus),
    )

    rows = []
    for season, g in df.groupby("season"):
        picks = {
            ("improve", "model"): g.nlargest(top_n, "p_improve"),
            ("improve", "naive"): g.nlargest(top_n, "naive"),
            ("decline", "model"): g.nsmallest(top_n, "p_improve"),
            ("decline", "naive"): g.nsmallest(top_n, "naive"),
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
    df: pd.DataFrame, stat: str, versus: str = "ytd",
    bins: tuple[float, ...] = (0, 0.2, 0.4, 0.6, 0.8, 1),
) -> pd.DataFrame:
    """Predicted chance of beating the reference vs the share that actually did."""
    df = scoreable(df, stat)
    p = improvement_probability(df, stat, df[PA_LEFT_COLUMN[stat]], versus)
    improved = DIRECTION[stat] * (df[f"actual_{stat}"] - reference(df, stat, versus)) > 0
    table = pd.DataFrame({"p": p, "improved": improved,
                          "bin": pd.cut(p, bins, include_lowest=True)})
    return (table.groupby("bin", observed=True)
            .agg(hitters=("p", "size"), predicted=("p", "mean"), actual=("improved", "mean"))
            .reset_index())
