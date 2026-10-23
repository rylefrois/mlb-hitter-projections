"""Explore rest-of-season forecasts from the backtest.

Run from the repository root, after building the data:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import altair as alt
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from mlb_ros_projections.backtest import EVAL_SEASONS
from mlb_ros_projections.calibration import interval
from mlb_ros_projections.models import FINAL_MODEL, sd_from_parts
from mlb_ros_projections.trends import (
    list_track_record,
    probability_calibration,
    rank_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "mlb.duckdb"
REPORTS_DIR = ROOT / "reports"

STATS = {
    "woba": {"label": "wOBA", "fmt": "{:.3f}", "trials": "woba_denom_ytd"},
    "k_rate": {"label": "K%", "fmt": "{:.1%}", "trials": "pa_ytd"},
    "ubb_rate": {"label": "BB% (unintentional)", "fmt": "{:.1%}", "trials": "pa_ytd"},
}
FORECAST_NAMES = {
    "league_average": "League average",
    "season_to_date": "Season to date",
    "marcel": "Marcel (preseason)",
    "eb_basic": "EB: basic",
    "eb_history": "EB: + history",
    "eb_full": "EB: + xwOBA & environment",
    "eb_playing_time": "EB: + playing time",
    "eb_park": "EB: + parks (final)",
}


def fmt(stat: str, value: float) -> str:
    return "–" if pd.isna(value) else STATS[stat]["fmt"].format(value)


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        df = con.execute("SELECT * FROM app_forecasts").df()
    df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.date
    return df


@st.cache_data
def load_report(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / name)


def ranges(row: pd.Series, stat: str, remaining_pa: int) -> dict[float, tuple[float, float]]:
    """80% and 95% ranges for the stat over the next `remaining_pa` PA."""
    sd = sd_from_parts(
        row[f"{stat}_per_pa_var"], row[f"{stat}_evidence"], remaining_pa,
        row[f"{stat}_interval_scale"],
    )
    out = {}
    for level in (0.8, 0.95):
        lower, upper = interval(pd.Series([row[stat]]), pd.Series([sd]), level)
        out[level] = (lower[0], upper[0])
    return out


def season_chart(history: pd.DataFrame, stat: str) -> alt.Chart:
    """Forecast with its 80% range across the season, against the baselines and outcomes."""
    # Tick only at the forecast dates, so each date is labeled exactly once.
    ticks = [alt.DateTime(year=d.year, month=d.month, date=d.day) for d in history["asof_date"]]
    x = alt.X("date:T", title="Forecast date", axis=alt.Axis(values=ticks, format="%b %-d"))
    rows = []
    for _, r in history.iterrows():
        lo, hi = ranges(r, stat, int(r["pa_ros"]) or 1)[0.8]
        rows.append({"date": pd.Timestamp(r["asof_date"]), "lower": lo, "upper": hi,
                     "Forecast": r[stat], "Season to date": r[f"ytd_{stat}"],
                     "Marcel": r[f"marcel_{stat}"], "What happened next": r[f"actual_{stat}"]})
    data = pd.DataFrame(rows)
    long = data.melt(id_vars=["date"], value_vars=[
        "Forecast", "Season to date", "Marcel", "What happened next",
    ], var_name="series", value_name="value")

    axis_format = ".3f" if stat == "woba" else ".0%"
    color = alt.Color("series:N", title=None, scale=alt.Scale(
        domain=["Forecast", "Season to date", "Marcel", "What happened next"],
        range=["#2563eb", "#9ca3af", "#f59e0b", "#16a34a"],
    ), legend=alt.Legend(orient="bottom"))

    band = alt.Chart(data).mark_area(opacity=0.18, color="#2563eb").encode(
        x=x,
        y=alt.Y("lower:Q", title=STATS[stat]["label"], axis=alt.Axis(format=axis_format)),
        y2="upper:Q",
    )
    lines = alt.Chart(long).mark_line(point=True).encode(
        x=x,
        y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
        color=color,
        strokeDash=alt.condition(
            alt.datum.series == "Marcel", alt.value([5, 4]), alt.value([1, 0])
        ),
        tooltip=["series:N", alt.Tooltip("date:T", format="%b %d"),
                 alt.Tooltip("value:Q", format=axis_format)],
    )
    return (band + lines).properties(height=340)


ALL_TEAMS = "All teams"


def hitter_options(season_df: pd.DataFrame, team: str) -> dict[int, str]:
    """Hitters to choose from, as {batter id: display name}, most playing time first.

    A hitter traded midseason is listed under every team they played for, and
    always shows their full season. Duplicate names get their team added.
    """
    if team != ALL_TEAMS:
        on_team = season_df.loc[season_df["team"] == team, "batter"].unique()
        season_df = season_df[season_df["batter"].isin(on_team)]
    hitters = (
        season_df.sort_values("asof_date")
        .groupby("batter")
        .agg(name=("name", "first"), team=("team", "last"),
             pa_ytd=("pa_ytd", "first"), pa_ros=("pa_ros", "first"))
    )
    hitters["total_pa"] = hitters["pa_ytd"] + hitters["pa_ros"]
    hitters = hitters.sort_values("total_pa", ascending=False)
    duplicate = hitters["name"].duplicated(keep=False)
    hitters["display"] = hitters["name"].where(
        ~duplicate, hitters["name"] + " (" + hitters["team"] + ")"
    )
    return hitters["display"].to_dict()


def player_tab(df: pd.DataFrame, season: int, stat: str) -> None:
    season_df = df[df["season"] == season]
    teams = sorted(season_df["team"].dropna().unique())
    team_col, hitter_col, date_col = st.columns([1, 2, 1])
    # Keys include the season (and team), so these reset when it changes instead of
    # keeping a value, like a date from another season, that is no longer an option.
    team = team_col.selectbox("Team", [ALL_TEAMS, *teams], key=f"player_team_{season}")
    options = hitter_options(season_df, team)
    batter = hitter_col.selectbox("Hitter", list(options), format_func=options.get,
                                  key=f"player_hitter_{season}_{team}",
                                  help="Type to search. Sorted by playing time.")
    history = season_df[season_df["batter"] == batter].sort_values("asof_date")
    dates = history["asof_date"].tolist()
    asof = date_col.select_slider("Forecast date", options=dates, value=dates[0],
                                  key=f"player_date_{season}_{batter}",
                                  format_func=lambda d: d.strftime("%b %-d"))
    row = history[history["asof_date"] == asof].iloc[0]

    st.subheader(f"{row['name']} · {row['team']} · age {int(row['age'])}")
    st.caption(f"As of {asof:%B %-d, %Y}: {int(row['pa_ytd'])} PA so far this season, "
               f"{int(row['pa_ros'])} PA the rest of the way.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast", fmt(stat, row[stat]),
              delta=f"{row[stat] - row[f'marcel_{stat}']:+.3f} vs Marcel" if stat == "woba"
              else f"{row[stat] - row[f'marcel_{stat}']:+.1%} vs Marcel",
              delta_color="off")
    c2.metric("Marcel (preseason)", fmt(stat, row[f"marcel_{stat}"]))
    c3.metric("Season to date", fmt(stat, row[f"ytd_{stat}"]))
    c4.metric("What happened next", fmt(stat, row[f"actual_{stat}"]))

    remaining = st.slider(
        "Range over the next N plate appearances", 25, 700,
        value=int(np.clip(row["pa_ros"], 25, 700)), step=25,
        help="Defaults to the PA the hitter actually had left. Fewer PA means more luck, "
             "so a wider range.",
    )
    r = ranges(row, stat, remaining)
    st.markdown(
        f"**80% range:** {fmt(stat, r[0.8][0])} to {fmt(stat, r[0.8][1])} &nbsp;·&nbsp; "
        f"**95% range:** {fmt(stat, r[0.95][0])} to {fmt(stat, r[0.95][1])}"
    )

    st.altair_chart(season_chart(history, stat), width="stretch")
    st.caption("Shaded band: 80% range over the PA the hitter actually had left. "
               "Each point uses only data from before that date.")

    with st.expander("Why this forecast?", expanded=True):
        trials = row[STATS[stat]["trials"]]
        evidence = row[f"{stat}_evidence"]
        prior_strength = evidence - trials
        history_share = row["marcel_weighted_pa"] / (row["marcel_weighted_pa"] + 1200)
        park_change = row["pf_ros_woba"] / row["pf_hist_woba"] - 1
        st.markdown(f"""
- **This season vs. track record:** this season's {int(trials)} PA count for
  **{trials / evidence:.0%}** of the talent estimate. The Marcel projection counts
  as {prior_strength:.0f} PA of evidence, the other **{prior_strength / evidence:.0%}**.
- **Track record:** Marcel reliability {history_share:.2f} (0 = no MLB history,
  about 0.85 = an established regular). More history makes Marcel count for more.
- **Playing time:** {min(row['pa_ytd_share'], 1.5):.0%} of an everyday player's PA so far.
  Hitters with less playing time start from a slightly lower baseline.
- **Parks:** the team's remaining schedule is {row['pf_ros_woba'] - 1:+.1%} for wOBA,
  vs. {row['pf_hist_woba'] - 1:+.1%} for the parks behind the Marcel projection
  ({park_change:+.1%} change).
""")


def candidate_table(picks: pd.DataFrame, stat: str, chance_label: str, flip: bool,
                    reveal: bool) -> pd.DataFrame:
    """A ranked list of hitters, formatted for display."""
    chance = 1 - picks["p_improve"] if flip else picks["p_improve"]
    table = pd.DataFrame({
        "Hitter": picks["name"],
        "Team": picks["team"],
        "PA so far": picks["pa_ytd"].astype(int),
        "So far": picks[f"ytd_{stat}"].map(lambda v: fmt(stat, v)),
        "Forecast": picks[stat].map(lambda v: fmt(stat, v)),
        chance_label: chance.map("{:.0%}".format),
        "Why": picks["reasons"],
    })
    if reveal:
        # Released or sent down hitters have no outcome to show.
        table["What happened next"] = [
            fmt(stat, actual) if pa_left > 0 else "no more PA"
            for actual, pa_left in zip(picks[f"actual_{stat}"], picks["pa_ros"])
        ]
    return table


def trends_tab(df: pd.DataFrame, season: int, stat: str) -> None:
    st.markdown(
        "Which hitters' numbers so far misstate their real level? **Slumps to ignore** are "
        "hitters likely to do better from here; **hot starts to doubt** are likely to do worse. "
        "The chance is how often the next N PA should beat the hitter's season-to-date number, "
        "based on the forecast ranges."
    )
    season_df = df[df["season"] == season]
    dates = sorted(season_df["asof_date"].unique())
    teams = sorted(season_df["team"].dropna().unique())
    c1, c2, c3, c4 = st.columns(4)
    asof = c1.select_slider("Forecast date", options=dates, value=dates[1] if len(dates) > 1
                            else dates[0], format_func=lambda d: d.strftime("%b %-d"),
                            key=f"trends_date_{season}")
    team = c2.selectbox("Team", [ALL_TEAMS, *teams], key=f"trends_team_{season}")
    min_pa = c3.slider("Minimum PA so far", 25, 300, 75, step=25, key="trends_min_pa")
    next_pa = c4.slider("Next N plate appearances", 50, 400, 150, step=25, key="trends_next_pa",
                        help="Over more PA, luck evens out, so the calls get more confident.")

    pool = season_df[(season_df["asof_date"] == asof) & (season_df["pa_ytd"] >= min_pa)]
    if team != ALL_TEAMS:
        pool = pool[pool["team"] == team]
    if pool.empty:
        st.info("No hitters match these filters.")
        return
    reveal = st.toggle("Reveal what happened next", value=False,
                       help="These are past seasons, so you can check the calls.")

    improvers, decliners = rank_candidates(pool, stat, next_pa)
    left, right = st.columns(2)
    left.markdown("#### Slumps to ignore")
    left.dataframe(candidate_table(improvers, stat, "Chance to improve", False, reveal),
                   hide_index=True, width="stretch")
    right.markdown("#### Hot starts to doubt")
    right.dataframe(candidate_table(decliners, stat, "Chance to decline", True, reveal),
                    hide_index=True, width="stretch")

    st.divider()
    st.markdown(f"#### How have these lists done? Every season, as of {asof:%B %-d}")
    st.caption(
        f"All teams, hitters with {min_pa}+ PA so far and 50+ PA left. The naive list is "
        "simply the 10 worst (or best) hitters so far: bad starts improve partly by "
        "regression to the mean alone, so the model has to beat it to add anything."
    )
    same_date = df[(pd.to_datetime(df["asof_date"]).dt.month == asof.month)
                   & (df["pa_ytd"] >= min_pa)]
    record = list_track_record(same_date, stat)
    summary = record.groupby(["list", "method"])[["hit_rate", "avg_actual_change"]].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Slumps that improved: model", f"{summary.loc[('improve', 'model'), 'hit_rate']:.0%}")
    m2.metric("Slumps that improved: naive", f"{summary.loc[('improve', 'naive'), 'hit_rate']:.0%}")
    m3.metric("Hot starts that declined: model",
              f"{summary.loc[('decline', 'model'), 'hit_rate']:.0%}")
    m4.metric("Hot starts that declined: naive",
              f"{summary.loc[('decline', 'naive'), 'hit_rate']:.0%}")

    by_season = record.pivot_table(index="season", columns=["list", "method"], values="hit_rate")
    by_season.columns = [f"{'Slumps' if lst == 'improve' else 'Hot starts'}: {method}"
                         for lst, method in by_season.columns]
    with st.expander("Season by season"):
        st.dataframe(by_season.style.format("{:.0%}"), width="stretch")

    st.markdown("#### Are the chances honest?")
    st.caption("Every hitter on this date in every season, grouped by the predicted chance "
               "of improving. Points on the dashed line mean the chances were right.")
    calib = probability_calibration(same_date, stat)
    points = alt.Chart(calib).mark_circle(color="#2563eb").encode(
        x=alt.X("predicted:Q", title="Predicted chance of improving",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        y=alt.Y("actual:Q", title="Share that actually improved",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        size=alt.Size("hitters:Q", title="Hitters"),
        tooltip=[alt.Tooltip("predicted:Q", format=".0%"), alt.Tooltip("actual:Q", format=".0%"),
                 "hitters:Q"],
    )
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], color="#9ca3af").encode(x="x:Q", y="y:Q")
    st.altair_chart((diagonal + points).properties(height=320), width="stretch")


def league_tab(df: pd.DataFrame, season: int, stat: str) -> None:
    season_df = df[df["season"] == season]
    dates = sorted(season_df["asof_date"].unique())
    left, right = st.columns(2)
    asof = left.select_slider("Forecast date", options=dates, value=dates[0],
                              format_func=lambda d: d.strftime("%b %-d"), key=f"league_date_{season}")
    min_pa = right.slider("Minimum PA so far", 0, 300, 50, step=10)

    day = season_df[(season_df["asof_date"] == asof) & (season_df["pa_ytd"] >= min_pa)
                    & (season_df["pa_ros"] >= 50)].copy()
    if day.empty:
        st.info("No hitters match these filters.")
        return
    day["change"] = day[stat] - day[f"marcel_{stat}"]
    day["model_error"] = (day[stat] - day[f"actual_{stat}"]).abs()
    day["ytd_error"] = (day[f"ytd_{stat}"] - day[f"actual_{stat}"]).abs()

    st.markdown(f"#### Biggest moves from Marcel on {asof:%B %-d}")
    st.caption("Where this season's data moved the forecast most. Was the move right? "
               "Compare the forecast and the season-to-date line with what happened next.")
    top = day.reindex(day["change"].abs().sort_values(ascending=False).index).head(25)
    table = pd.DataFrame({
        "Hitter": top["name"],
        "Team": top["team"],
        "PA so far": top["pa_ytd"].astype(int),
        "Season to date": top[f"ytd_{stat}"].map(lambda v: fmt(stat, v)),
        "Marcel": top[f"marcel_{stat}"].map(lambda v: fmt(stat, v)),
        "Forecast": top[stat].map(lambda v: fmt(stat, v)),
        "What happened next": top[f"actual_{stat}"].map(lambda v: fmt(stat, v)),
        "Closer": np.where(top["model_error"] < top["ytd_error"], "Forecast", "Season to date"),
    })
    st.dataframe(table, hide_index=True, width="stretch")

    weights = day["woba_denom_ros" if stat == "woba" else "pa_ros"]

    def rmse(col):
        return np.sqrt(np.average((day[col] - day[f"actual_{stat}"]) ** 2, weights=weights))

    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast error (RMSE)", fmt(stat, rmse(stat)))
    c2.metric("Marcel error", fmt(stat, rmse(f"marcel_{stat}")))
    c3.metric("Season-to-date error", fmt(stat, rmse(f"ytd_{stat}")))
    st.caption(f"All {len(day)} hitters on this date with at least {min_pa} PA so far and "
               "50 PA left, weighted by PA left.")


def performance_tab(stat: str) -> None:
    first, last = min(EVAL_SEASONS), max(EVAL_SEASONS)
    st.markdown(f"#### Accuracy across {first}–{last} (2020 excluded)")
    st.caption("RMSE of rest-of-season forecasts, weighted by PA left. Lower is better. "
               "The noise floor is the error even perfect knowledge of talent would leave.")
    overall = load_report("backtest_overall.csv")
    table = overall.pivot(index="forecast", columns="stat", values="rmse")
    table = table.reindex([f for f in FORECAST_NAMES if f in table.index])
    floor = overall.groupby("stat")["noise_floor"].first()
    table.loc["noise_floor"] = floor
    table.index = [FORECAST_NAMES.get(i, "Noise floor") for i in table.index]
    table = table[list(STATS)].rename(columns={s: STATS[s]["label"] for s in STATS})
    st.dataframe(table.style.format("{:.4f}"), width="stretch")

    st.markdown("#### Are the ranges honest?")
    st.caption(f"Share of actual results inside each range ({FORECAST_NAMES[FINAL_MODEL]}). "
               "A well-calibrated 80% range contains about 80%.")
    cal = load_report("calibration_overall.csv")
    cal = cal.pivot(index="level", columns="stat", values="inside")[list(STATS)]
    cal.index = [f"{level:.0%} range" for level in cal.index]
    cal = cal.rename(columns={s: STATS[s]["label"] for s in STATS})
    st.dataframe(cal.style.format("{:.1%}"), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Rest-of-Season Projections", layout="wide")
    st.title("Rest-of-season hitter projections")
    st.caption("Marcel updated with this season's data (empirical Bayes), with honest "
               "uncertainty ranges. Every forecast uses only data available on its date.")

    if not DB_PATH.exists():
        st.error("No data found. Build it first: see the README.")
        st.stop()
    df = load_forecasts()

    with st.sidebar:
        season = st.selectbox("Season", sorted(df["season"].unique(), reverse=True))
        stat = st.radio("Stat", list(STATS), format_func=lambda s: STATS[s]["label"])

    player, trends, league, performance = st.tabs(
        ["Player", "Slumps & hot starts", "League", "Model performance"]
    )
    with player:
        player_tab(df, season, stat)
    with trends:
        trends_tab(df, season, stat)
    with league:
        league_tab(df, season, stat)
    with performance:
        performance_tab(stat)


main()
