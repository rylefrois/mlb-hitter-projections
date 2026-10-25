# Model card: rest-of-season hitter projections

## What it does

Given any date in the season, the model forecasts each MLB hitter's **rate for the rest of the season** in three stats, with a range around each forecast:

| Stat | Definition |
|---|---|
| **wOBA** | Weighted on-base average, on Statcast's scale |
| **K%** | Strikeouts per plate appearance |
| **BB%** | Unintentional walks per plate appearance |

The ranges are designed so that an 80% range contains the actual result about 80% of the time.

**Intended use:** deciding how much to believe a hitter's numbers so far. Is a slump or a hot start real, and has their outlook changed since spring?

**Not intended for:** forecasting playing time, injuries, or counting stats such as home runs; pitchers; or any season before 2018.

## How it works

The model is **Marcel, updated with this season's data** using empirical Bayes:

```
forecast = (this season's events + strength × prior) / (this season's PA + strength)
```

1. **Prior (preseason):** Marcel (Tango): the last three seasons weighted 5/4/3, regressed toward league average with 1,200 PA, and adjusted for age. Several adjustments are then applied to it:
   - **Run environment:** shifted when this season's league rates differ from the same date in recent seasons.
   - **Playing time:** shifted for hitters with little MLB history or a small share of playing time. Those hitters are below average as a group.
   - **Parks:** put in neutral-park terms using the parks behind the hitter's history.
2. **Strength:** how many plate appearances of evidence the prior is worth. It is lower for hitters with little MLB history. It is learned separately for each stat, because stats settle down at different speeds.
3. **This season:** results so far, in neutral-park terms. For wOBA, actual results are blended with **xwOBA** (expected wOBA from exit velocity and launch angle), which is less noisy. xwOBA is computed without regard to park, so it isn't park-adjusted.
4. **Back to real parks:** the neutral forecast is adjusted for the parks on the team's **remaining schedule**.
5. **Range:** the spread combines two sources of uncertainty:
   - **Talent uncertainty:** per-PA variance ÷ (this season's PA + strength).
   - **Luck over the PA still to come:** per-PA variance ÷ PA left.

   A width correction, learned per stat, scales the result so the ranges are calibrated.

### Learned parameters

Each forecast season is fit only on earlier seasons. The ranges below are the values across the 8 forecast seasons.

| | K% | BB% | wOBA |
|---|---|---|---|
| Prior worth, no MLB history (PA) | 51–96 | 114–162 | 176–256 |
| Prior worth, established regular (PA) | 324–384 | 587–716 | 569–616 |
| xwOBA share of this season's signal | | | 0.56–0.71 |
| Park weight (1 = full park factor) | 0.18–0.60 | 0.00–0.35 | 0.29–0.50 |
| Range width correction | 1.09–1.13 | 1.12–1.14 | 0.99–1.00 |

What the parameters show:
- **K% reflects a hitter fastest.** An established hitter's track record is worth about 330 PA of current-season evidence for K%, against about 600 for wOBA and 700 for BB%.
- **The best read of current wOBA is about two-thirds xwOBA.**
- **Park factors are only trusted at roughly half strength,** because three seasons of park data are still noisy.
- **K% and BB% vary about 12% more than pure luck would explain,** which is what the width corrections above 1 show. True talent likely drifts during a season.

## Data

| Source | What | Seasons |
|---|---|---|
| Baseball Savant (Statcast), via `pybaseball` | 8.15M regular-season pitches → 2.09M plate appearances | 2015–2026 |
| Chadwick Bureau register | Player IDs, names, birthdates | All |
| MLB Stats API | The ballpark of every game | 2015–2026 |

Ballparks are keyed on the venue's ID, which stays the same when a park is renamed. This means **team relocations** are handled correctly: in 2025 the Athletics played in Sacramento and the Rays at Steinbrenner Field. So are **neutral-site games** such as Tokyo, London and the Bristol speedway game.

## How it was validated

**Backtest:** 15,108 hitter-snapshots. Each snapshot is one hitter on one forecast date: May 1, Jun 1, Jul 1, Aug 1 or Sep 1. Seasons 2018–2026 are used; 2020 is excluded because it started in late July and had only 60 games. 2015–2017 serve as history for the earliest forecasts.

**No look-ahead:**
- Every input uses only games before the forecast date.
- Park factors and league environment use earlier seasons only.
- Every parameter, including the range width correction, is **forward-chained**: each season is forecast with parameters fit only on earlier seasons.

**Scoring:**
- RMSE of each forecast against the actual rest-of-season rate, weighted by plate appearances left.
- Rows count when the hitter has at least 1 PA so far and at least 50 PA left.
- Every forecast is scored on exactly the same rows, and a forecast that is missing any row is rejected.

### Accuracy (RMSE; lower is better)

| Forecast | K% | BB% | wOBA |
|---|---|---|---|
| Season to date | .0644 | .0402 | .0742 |
| League average | .0657 | .0322 | .0471 |
| Marcel (preseason) | .0466 | .0257 | .0432 |
| + this season's data | .0419 | .0247 | .0423 |
| + history-dependent prior | .0412 | .0246 | .0422 |
| + xwOBA and run environment | .0413 | .0246 | .0415 |
| + playing time | **.0411** | .0245 | .0414 |
| **+ parks (final)** | .0412 | **.0245** | **.0413** |
| *Noise floor* | *.0287* | *.0188* | *.0356* |

The **noise floor** is the error a forecaster would still make if they knew every hitter's true talent exactly, because a few hundred plate appearances are partly luck.

**Final model compared with Marcel:**
- **Squared error is 22% lower for K%, 10% for BB% and 9% for wOBA.**
- **That closes 35% (K%), 20% (BB%) and 27% (wOBA) of the gap between Marcel and the noise floor.**

**The final model beats Marcel and season-to-date at every forecast date and every sample size.** For example, for wOBA on May 1 its error is .0363, against .0377 for Marcel and .0835 for season to date.

### Calibration (share of actual results inside each range)

| Range | K% | BB% | wOBA |
|---|---|---|---|
| 50% | 49.5% | 50.5% | 49.8% |
| 80% | 79.8% | 81.3% | 80.5% |
| 95% | 95.1% | 95.5% | 95.1% |

**Bias** is the forecast minus the actual result, weighted by PA left:

| Stat | Bias |
|---|---|
| K% | +0.43 pts |
| BB% | +0.10 pts |
| wOBA | −.0003 |

## Limitations

- **Playing time isn't forecast.** A range is for "the hitter's next N PA". The backtest uses the number of PA the hitter actually got, and the app lets the user choose N.
- **The evaluation only covers hitters with 50+ PA left.** Hitters who were released or injured early drop out.
- **K% forecasts run about 0.4 pts high,** almost all of it in **2021–2023**: +1.6, +0.8 and +0.4 pts. Every other season is within 0.2 pts. Marcel's errors those seasons were smaller (+0.5, +0.4, −0.2), so this comes from the model's adjustments, not from the prior. A plausible cause, not yet tested: the league strikeout rate rose every year through 2020–21 and then fell, and adjustments learned during the rising years didn't anticipate the turn. It's a league-wide, season-level error, not a problem with particular hitters.
- **Talent is treated as fixed within a season.** Injuries, swing changes and role changes are not modeled. The width correction makes the ranges honest about this, but the center of the forecast doesn't react to these changes.
- **Parks:**
  - **New ballparks** are treated as neutral in their first season.
  - **Changes to park dimensions** take up to three seasons to show fully in the factors.
  - **Park factors are averages** over both teams and over left- and right-handed hitters.
- **wOBA uses Statcast's scale** (fixed weights; reaching on an error counts like a single). It runs a few points higher than FanGraphs' wOBA, but it is consistent across seasons.
- **"Chance to beat" percentages in the app:**
  - Against **season to date**, they are within about 5 points of what actually happened.
  - Against **the preseason projection**, they run optimistic for wOBA and BB%. For example, wOBA hitters given 48% beat their projection 43% of the time, and hitters given 66% did 61% of the time. For K% they run slightly pessimistic: hitters given 68% did it 71% of the time. Use them for ranking, not as exact odds.
- **The team on a forecast date** is the one the hitter last batted for, so a trade shows up from the hitter's first game with the new team.
