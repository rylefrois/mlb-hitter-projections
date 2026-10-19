-- Inputs for the empirical-Bayes model (models.py).
-- Needs: batter_snapshot (features.sql), marcel (baselines.sql).


-- League rates from opening day up to each forecast date.
CREATE OR REPLACE TABLE league_snapshot AS
SELECT
    season,
    asof_date,
    sum(k_ytd)   / sum(pa_ytd)          AS k_rate,
    sum(ubb_ytd) / sum(pa_ytd)          AS ubb_rate,
    sum(woba_num_ytd) / sum(woba_denom_ytd) AS woba
FROM batter_snapshot
GROUP BY season, asof_date
HAVING sum(pa_ytd) > 0;


-- Run environment: this season's league rates so far, relative to the same
-- point in the previous three seasons. Comparing like with like (May 1 with
-- May 1) keeps normal seasonal swings, like cold Aprils, out of the factor.
-- 2020 is skipped as a comparison season: it started in late July.
CREATE OR REPLACE TABLE environment AS
SELECT
    c.season,
    c.asof_date,
    c.k_rate   / avg(p.k_rate)   AS k_rate_factor,
    c.ubb_rate / avg(p.ubb_rate) AS ubb_rate_factor,
    c.woba     / avg(p.woba)     AS woba_factor
FROM league_snapshot AS c
JOIN league_snapshot AS p
  ON p.season BETWEEN c.season - 3 AND c.season - 1
 AND p.season <> 2020
 AND month(p.asof_date) = month(c.asof_date)
GROUP BY c.season, c.asof_date, c.k_rate, c.ubb_rate, c.woba;


-- How much a single PA's wOBA value varies, by season. Forecast ranges use
-- the previous season's value, which is known at forecast time.
CREATE OR REPLACE TABLE woba_pa_variance AS
SELECT season, var_pop(woba_value) AS woba_pa_var
FROM pa
WHERE woba_denom = 1
GROUP BY season;


-- One row per hitter and forecast date: this season's counts so far, the
-- Marcel prior, the environment factors, and the rest-of-season actuals.
CREATE OR REPLACE TABLE model_inputs AS
SELECT
    s.season,
    s.asof_date,
    s.batter,

    s.pa_ytd,
    s.k_ytd,
    s.ubb_ytd,
    s.woba_denom_ytd,
    s.woba_num_ytd,
    s.xwoba_num_ytd,

    m.marcel_weighted_pa,
    m.marcel_k_rate,
    m.marcel_ubb_rate,
    m.marcel_woba,

    -- No comparison seasons (e.g. 2016 has only 2015): assume no change.
    coalesce(e.k_rate_factor, 1)   AS k_rate_factor,
    coalesce(e.ubb_rate_factor, 1) AS ubb_rate_factor,
    coalesce(e.woba_factor, 1)     AS woba_factor,

    v.woba_pa_var AS prev_woba_pa_var,

    s.pa_ros,
    s.woba_denom_ros,
    s.k_ros   / nullif(s.pa_ros, 0)              AS actual_k_rate,
    s.ubb_ros / nullif(s.pa_ros, 0)              AS actual_ubb_rate,
    s.woba_num_ros / nullif(s.woba_denom_ros, 0) AS actual_woba
FROM batter_snapshot AS s
JOIN marcel AS m USING (season, batter)
LEFT JOIN environment AS e USING (season, asof_date)
LEFT JOIN woba_pa_variance AS v ON v.season = s.season - 1;
