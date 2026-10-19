-- Baseline forecasts the main model has to beat.
-- Needs: batter_season and batter_snapshot (features.sql), people (register).


-- League-wide rates per season.
CREATE OR REPLACE TABLE league_season AS
SELECT
    season,
    sum(k)   / sum(pa)          AS k_rate,
    sum(ubb) / sum(pa)          AS ubb_rate,
    sum(woba_num)  / sum(woba_denom) AS woba,
    sum(xwoba_num) / sum(woba_denom) AS xwoba
FROM batter_season
GROUP BY season;


-- Each hitter's age for a season, using the baseball convention of age on June 30.
CREATE OR REPLACE TABLE batter_age AS
SELECT
    b.season,
    b.batter,
    b.season - p.birth_year
        - CASE WHEN (p.birth_month, p.birth_day) > (6, 30) THEN 1 ELSE 0 END AS age
FROM batter_season AS b
JOIN people AS p ON p.key_mlbam = b.batter;


-- Marcel (Tom Tango's deliberately simple projection system), preseason version:
--   1. Weight the last three seasons 5/4/3, most recent first.
--   2. Add 1200 weighted PA of league-average performance (regression to the mean).
--   3. Adjust for age: +0.6% per year under 29, -0.3% per year over 29.
-- Only data from before the target season is used.

-- League average to regress toward: the same 5/4/3 weighting of league rates.
CREATE OR REPLACE TABLE marcel_league AS
SELECT
    t.season,
    sum(w * l.k_rate)   / sum(w) AS k_rate,
    sum(w * l.ubb_rate) / sum(w) AS ubb_rate,
    sum(w * l.woba)     / sum(w) AS woba,
    sum(w * l.xwoba)    / sum(w) AS xwoba
FROM (SELECT DISTINCT season FROM batter_season) AS t
JOIN league_season AS l
  ON l.season BETWEEN t.season - 3 AND t.season - 1
CROSS JOIN LATERAL (SELECT 6 - (t.season - l.season) AS w)  -- 1 year back: 5, 2: 4, 3: 3
GROUP BY t.season;


CREATE OR REPLACE TABLE marcel AS
WITH history AS (
    SELECT
        t.season,
        t.batter,
        coalesce(sum(w * h.pa), 0)         AS wpa,
        coalesce(sum(w * h.k), 0)          AS wk,
        coalesce(sum(w * h.ubb), 0)        AS wubb,
        coalesce(sum(w * h.woba_denom), 0) AS wden,
        coalesce(sum(w * h.woba_num), 0)   AS wwoba,
        coalesce(sum(w * h.xwoba_num), 0)  AS wxwoba
    FROM batter_season AS t
    LEFT JOIN batter_season AS h
      ON h.batter = t.batter
     AND h.season BETWEEN t.season - 3 AND t.season - 1
    CROSS JOIN LATERAL (SELECT 6 - (t.season - h.season) AS w)
    GROUP BY t.season, t.batter
),

regressed AS (
    SELECT
        h.season,
        h.batter,
        h.wpa,
        (h.wk    + 1200 * lg.k_rate)   / (h.wpa  + 1200) AS k_rate,
        (h.wubb  + 1200 * lg.ubb_rate) / (h.wpa  + 1200) AS ubb_rate,
        (h.wwoba  + 1200 * lg.woba)    / (h.wden + 1200) AS woba,
        (h.wxwoba + 1200 * lg.xwoba)   / (h.wden + 1200) AS xwoba
    FROM history AS h
    JOIN marcel_league AS lg USING (season)
)

SELECT
    r.season,
    r.batter,
    a.age,
    r.wpa AS marcel_weighted_pa,
    -- Strikeouts are bad for a hitter, so improving with age means fewer of them:
    -- divide by the age factor instead of multiplying.
    r.k_rate   / m.age_factor AS marcel_k_rate,
    r.ubb_rate * m.age_factor AS marcel_ubb_rate,
    r.woba     * m.age_factor AS marcel_woba,
    r.xwoba    * m.age_factor AS marcel_xwoba
FROM regressed AS r
LEFT JOIN batter_age AS a USING (season, batter)
CROSS JOIN LATERAL (
    SELECT CASE
        WHEN a.age IS NULL THEN 1.0           -- unknown birthdate: no adjustment
        WHEN a.age > 29 THEN 1 - 0.003 * (a.age - 29)
        ELSE 1 + 0.006 * (29 - a.age)
    END AS age_factor
) AS m;


-- Both baselines lined up with the rest-of-season actuals for every snapshot.
CREATE OR REPLACE TABLE batter_baselines AS
SELECT
    s.season,
    s.asof_date,
    s.batter,
    s.pa_ytd,
    s.pa_ros,
    s.woba_denom_ytd,
    s.woba_denom_ros,

    -- Baseline 1: the hitter keeps doing exactly what they've done so far.
    s.k_ytd   / nullif(s.pa_ytd, 0)         AS ytd_k_rate,
    s.ubb_ytd / nullif(s.pa_ytd, 0)         AS ytd_ubb_rate,
    s.woba_num_ytd  / nullif(s.woba_denom_ytd, 0) AS ytd_woba,
    s.xwoba_num_ytd / nullif(s.woba_denom_ytd, 0) AS ytd_xwoba,

    -- Baseline 2: preseason Marcel, ignoring everything this season.
    m.marcel_k_rate,
    m.marcel_ubb_rate,
    m.marcel_woba,
    m.marcel_xwoba,

    -- What actually happened (the forecast target).
    s.k_ros   / nullif(s.pa_ros, 0)         AS actual_k_rate,
    s.ubb_ros / nullif(s.pa_ros, 0)         AS actual_ubb_rate,
    s.woba_num_ros / nullif(s.woba_denom_ros, 0) AS actual_woba
FROM batter_snapshot AS s
JOIN marcel AS m USING (season, batter);
