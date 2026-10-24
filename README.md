# Rest-of-season MLB hitter projections

How much should you believe a hitter's season so far? This project forecasts every MLB hitter's **rest-of-season wOBA, K% and BB%** from any date in the season, with ranges that are honest about uncertainty.

The method starts from Marcel (a simple, well-known projection system) and updates it with this season's data using empirical Bayes. It's backtested on **15,108 forecasts across 2018–2026** using only data available on each forecast date.

**Read next:**
- [Decision memo](docs/decision_memo.md): which slumps, hot starts and breakouts to believe.
- [Model card](docs/model_card.md): how the model works, how it was validated, and its limitations.

## Results

Error in forecasting each hitter's rest-of-season rate (RMSE, weighted by plate appearances left; lower is better):

| Forecast | K% | BB% | wOBA |
|---|---|---|---|
| Season to date | .0644 | .0402 | .0742 |
| League average for everyone | .0657 | .0322 | .0471 |
| Marcel (preseason projection) | .0466 | .0257 | .0432 |
| **This model** | **.0412** | **.0245** | **.0413** |
| *Noise floor (perfect knowledge of talent)* | *.0287* | *.0188* | *.0356* |

- **Squared error is 22%, 10% and 9% lower than Marcel's,** and the model beats both baselines at every forecast date and sample size.
- **Season-to-date wOBA predicts worse than assuming everyone is league average.**
- **The ranges are calibrated:** 80% ranges contain the actual result 79.8% (K%), 81.3% (BB%) and 80.5% (wOBA) of the time.
- **Every idea is scored on its own** in a step-by-step comparison (an ablation), including one that didn't help K%. See the [model card](docs/model_card.md#accuracy-rmse-lower-is-better).

## The app

`make app` opens a Streamlit dashboard with three tabs:

- **Player:** a hitter's forecast on any date, with 80% and 95% ranges over the next N plate appearances. It also shows how the forecast moved through the season, and a plain-language "why this forecast" breakdown: track record vs. this season, playing time and parks.
- **Hitters to watch:** ranked lists of whose numbers to believe. One view compares the forecast with **season to date** (slumps to ignore, hot starts to doubt). The other compares it with the **preseason projection** (outlooks raised or lowered). Each list comes with its track record in past seasons against a naive list, and a check of whether its stated chances came true.
- **Model performance:** accuracy by forecast date and sample size, plus calibration.

## How it works

```
Statcast pitches ──► plate appearances ──► "as of" snapshots (May 1 … Sep 1)
                                                │
Marcel prior (3 seasons, age) ──┐               ▼
playing-time & environment ─────┼──►  empirical-Bayes update  ──►  forecast + range
park factors (by ballpark) ─────┘     (+ xwOBA for wOBA)
```

- **No look-ahead:**
  - Inputs use only games before the forecast date.
  - Park factors and league environment use earlier seasons only.
  - All parameters are **forward-chained**: each season is forecast with parameters fit only on earlier seasons.
- **Ballparks, not teams:** park factors key on the actual venue, so the Athletics' and Rays' 2025 moves and neutral-site games are handled correctly.
- **Tested:** 64 tests. They cover the SQL logic, a simulation that checks the fitting recovers known values, calibration math, and the app itself.

## Quickstart

```bash
make install   # Python 3.11+, creates .venv
make all       # download data (a few hours the first time), build, backtest
make app       # open the dashboard
make test
```

`START` and `END` choose the seasons, e.g. `make data START=2023 END=2024`. The default is 2015–2026.

## Layout

```
src/mlb_ros_projections/
  ingest.py, register.py, venues.py   download Statcast, player register, ballparks
  build_db.py                         DuckDB views over the raw files
  features.sql                        plate appearances and as-of-date snapshots
  baselines.sql                       season-to-date and Marcel
  parks.sql                           park factors and each hitter's park exposure
  model_inputs.sql                    everything the model reads, one row per snapshot
  models.py                           the empirical-Bayes model and forward-chained fitting
  backtest.py, calibration.py         scoring, ranges, bias
  trends.py, insights.py              ranked lists and the analyses behind the memo
  forecasts.py                        saves the final forecasts for the app
app/streamlit_app.py                  the dashboard
reports/                              backtest and calibration results (CSV)
docs/                                 model card and decision memo
tests/
```

## Data and credits

- **Pitch data:** [Baseball Savant](https://baseballsavant.mlb.com) (Statcast), via [`pybaseball`](https://github.com/jldbc/pybaseball).
- **Player IDs and birthdates:** the [Chadwick Bureau register](https://github.com/chadwickbureau/register).
- **Ballparks:** the MLB Stats API schedule.
- **Marcel** is Tom Tango's projection system.

Raw data isn't included in this repository. `make data` downloads it.
