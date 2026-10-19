-- Builds hitter feature tables from the raw `pitches` view.
-- Run with: python -m mlb_ros_projections.features


-- One row per completed plate appearance.
-- The last pitch of a PA is the only row with `events` filled in.
-- `truncated_pa` means the PA never finished (e.g. a runner made the third
-- out mid at-bat), so it is not a real plate appearance.
CREATE OR REPLACE TABLE pa AS
SELECT
    year(game_date) AS season,
    game_date,
    game_pk,
    at_bat_number,
    batter,
    pitcher,
    stand,
    p_throws,
    home_team,
    CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS bat_team,
    events,
    events IN ('strikeout', 'strikeout_double_play') AS is_k,
    -- Unintentional walks only: an intentional walk is the manager's choice, not hitter skill.
    events = 'walk' AS is_ubb,
    -- wOBA leaves out intentional walks and sac bunts (denominator 0 or NULL).
    coalesce(woba_denom, 0) AS woba_denom,
    CASE WHEN woba_denom = 1 THEN woba_value ELSE 0 END AS woba_value,
    -- xwOBA is missing for a few batted balls with no tracking data; fall back to actual wOBA.
    CASE
        WHEN woba_denom = 1 THEN coalesce(estimated_woba_using_speedangle, woba_value)
        ELSE 0
    END AS xwoba_value
FROM pitches
WHERE events IS NOT NULL
  AND events <> 'truncated_pa';


-- Hitter totals per day. Snapshots are built from this smaller table.
CREATE OR REPLACE TABLE batter_day AS
SELECT
    season,
    game_date,
    batter,
    count(*) AS pa,
    sum(is_k::INT) AS k,
    sum(is_ubb::INT) AS ubb,
    sum(woba_denom) AS woba_denom,
    sum(woba_value) AS woba_num,
    sum(xwoba_value) AS xwoba_num
FROM pa
GROUP BY ALL;


-- Full-season hitter totals, used later for priors (e.g. Marcel).
CREATE OR REPLACE TABLE batter_season AS
SELECT
    season,
    batter,
    sum(pa) AS pa,
    sum(k) AS k,
    sum(ubb) AS ubb,
    sum(woba_denom) AS woba_denom,
    sum(woba_num) AS woba_num,
    sum(xwoba_num) AS xwoba_num
FROM batter_day
GROUP BY ALL;


-- The dates we forecast from: the first of May through September each season.
-- "As of" means at the start of that day, so games on asof_date count as future.
CREATE OR REPLACE TABLE snapshot_dates AS
SELECT DISTINCT
    p.season,
    make_date(p.season, m.month, 1) AS asof_date
FROM pa AS p
CROSS JOIN (SELECT unnest([5, 6, 7, 8, 9]) AS month) AS m;


-- For each hitter and forecast date: what was known (ytd = year to date)
-- and what happened afterward (ros = rest of season, the forecast target).
-- Splitting strictly on asof_date keeps future data out of the inputs.
CREATE OR REPLACE TABLE batter_snapshot AS
SELECT
    s.season,
    s.asof_date,
    d.batter,

    coalesce(sum(d.pa)         FILTER (WHERE d.game_date <  s.asof_date), 0) AS pa_ytd,
    coalesce(sum(d.k)          FILTER (WHERE d.game_date <  s.asof_date), 0) AS k_ytd,
    coalesce(sum(d.ubb)        FILTER (WHERE d.game_date <  s.asof_date), 0) AS ubb_ytd,
    coalesce(sum(d.woba_denom) FILTER (WHERE d.game_date <  s.asof_date), 0) AS woba_denom_ytd,
    coalesce(sum(d.woba_num)   FILTER (WHERE d.game_date <  s.asof_date), 0) AS woba_num_ytd,
    coalesce(sum(d.xwoba_num)  FILTER (WHERE d.game_date <  s.asof_date), 0) AS xwoba_num_ytd,

    coalesce(sum(d.pa)         FILTER (WHERE d.game_date >= s.asof_date), 0) AS pa_ros,
    coalesce(sum(d.k)          FILTER (WHERE d.game_date >= s.asof_date), 0) AS k_ros,
    coalesce(sum(d.ubb)        FILTER (WHERE d.game_date >= s.asof_date), 0) AS ubb_ros,
    coalesce(sum(d.woba_denom) FILTER (WHERE d.game_date >= s.asof_date), 0) AS woba_denom_ros,
    coalesce(sum(d.woba_num)   FILTER (WHERE d.game_date >= s.asof_date), 0) AS woba_num_ros,
    coalesce(sum(d.xwoba_num)  FILTER (WHERE d.game_date >= s.asof_date), 0) AS xwoba_num_ros
FROM snapshot_dates AS s
JOIN batter_day AS d USING (season)
GROUP BY ALL;
