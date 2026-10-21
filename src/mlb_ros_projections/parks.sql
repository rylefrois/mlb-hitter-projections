-- Park factors and each hitter's exposure to them.
-- Needs: pa, snapshot_dates (features.sql).
--
-- A park is identified by its home team's code. Games at neutral sites
-- (international series, special events) are counted at the nominal home
-- team's park, which adds a little noise.


-- Totals at each park, both teams combined.
CREATE OR REPLACE TABLE park_season AS
SELECT
    season,
    home_team AS park,
    count(*)          AS pa,
    sum(is_k::INT)    AS k,
    sum(is_ubb::INT)  AS ubb,
    sum(woba_denom)   AS woba_denom,
    sum(woba_value)   AS woba_num
FROM pa
GROUP BY ALL;


-- Totals in each team's road games, both teams combined.
CREATE OR REPLACE TABLE team_road_season AS
SELECT
    season,
    away_team AS team,
    count(*)          AS pa,
    sum(is_k::INT)    AS k,
    sum(is_ubb::INT)  AS ubb,
    sum(woba_denom)   AS woba_denom,
    sum(woba_value)   AS woba_num
FROM pa
GROUP BY ALL;


-- Park factor = rate at the park / rate in the same team's road games.
-- The same teams play on both sides, so team quality cancels out. Factors
-- for a season use only the previous three seasons (skipping 2020), so they
-- are known at forecast time. 1.10 means the park inflates that stat by 10%.
CREATE OR REPLACE TABLE park_factors AS
WITH target AS (
    SELECT DISTINCT season FROM pa
),
home AS (
    SELECT t.season, h.park,
           sum(h.pa) AS pa, sum(h.k) AS k, sum(h.ubb) AS ubb,
           sum(h.woba_denom) AS woba_denom, sum(h.woba_num) AS woba_num
    FROM target AS t
    JOIN park_season AS h
      ON h.season BETWEEN t.season - 3 AND t.season - 1
     AND h.season <> 2020
    GROUP BY t.season, h.park
),
road AS (
    SELECT t.season, r.team AS park,
           sum(r.pa) AS pa, sum(r.k) AS k, sum(r.ubb) AS ubb,
           sum(r.woba_denom) AS woba_denom, sum(r.woba_num) AS woba_num
    FROM target AS t
    JOIN team_road_season AS r
      ON r.season BETWEEN t.season - 3 AND t.season - 1
     AND r.season <> 2020
    GROUP BY t.season, r.team
)
SELECT
    season,
    park,
    home.pa AS home_pa,
    (home.k   / home.pa) / (road.k   / road.pa)                         AS k_rate,
    (home.ubb / home.pa) / (road.ubb / road.pa)                         AS ubb_rate,
    (home.woba_num / home.woba_denom) / (road.woba_num / road.woba_denom) AS woba
FROM home
JOIN road USING (season, park);


-- Average park factor over the PA each hitter has had so far this season.
-- Parks with no factor yet (e.g. 2015, nothing earlier) count as neutral.
CREATE OR REPLACE TABLE park_exposure_ytd AS
SELECT
    s.season,
    s.asof_date,
    p.batter,
    avg(coalesce(f.k_rate, 1))   AS pf_ytd_k_rate,
    avg(coalesce(f.ubb_rate, 1)) AS pf_ytd_ubb_rate,
    avg(coalesce(f.woba, 1))     AS pf_ytd_woba
FROM snapshot_dates AS s
JOIN pa AS p
  ON p.season = s.season
 AND p.game_date < s.asof_date
LEFT JOIN park_factors AS f
  ON f.season = p.season
 AND f.park = p.home_team
GROUP BY s.season, s.asof_date, p.batter;


-- Average park factor over the hitter's team's remaining schedule. The
-- schedule is published before the season, so this is known at forecast
-- time. The hitter's team is the one he last batted for before the date.
CREATE OR REPLACE TABLE park_exposure_ros AS
WITH team_at_date AS (
    SELECT s.season, s.asof_date, p.batter, arg_max(p.bat_team, p.game_date) AS team
    FROM snapshot_dates AS s
    JOIN pa AS p
      ON p.season = s.season
     AND p.game_date < s.asof_date
    GROUP BY s.season, s.asof_date, p.batter
),
games AS (
    SELECT DISTINCT season, game_pk, game_date, home_team, away_team FROM pa
),
team_remaining AS (
    SELECT
        s.season,
        s.asof_date,
        t.team,
        avg(coalesce(f.k_rate, 1))   AS pf_ros_k_rate,
        avg(coalesce(f.ubb_rate, 1)) AS pf_ros_ubb_rate,
        avg(coalesce(f.woba, 1))     AS pf_ros_woba
    FROM snapshot_dates AS s
    JOIN (SELECT DISTINCT season, home_team AS team FROM games) AS t
      ON t.season = s.season
    JOIN games AS g
      ON g.season = s.season
     AND g.game_date >= s.asof_date
     AND t.team IN (g.home_team, g.away_team)
    LEFT JOIN park_factors AS f
      ON f.season = g.season
     AND f.park = g.home_team
    GROUP BY s.season, s.asof_date, t.team
)
SELECT d.season, d.asof_date, d.batter, r.pf_ros_k_rate, r.pf_ros_ubb_rate, r.pf_ros_woba
FROM team_at_date AS d
JOIN team_remaining AS r USING (season, asof_date, team);


-- Park factor behind each hitter's Marcel projection: his PA over the
-- previous three seasons, weighted 5/4/3 like Marcel itself.
CREATE OR REPLACE TABLE park_exposure_hist AS
SELECT
    t.season,
    p.batter,
    sum(w * coalesce(f.k_rate, 1))   / sum(w) AS pf_hist_k_rate,
    sum(w * coalesce(f.ubb_rate, 1)) / sum(w) AS pf_hist_ubb_rate,
    sum(w * coalesce(f.woba, 1))     / sum(w) AS pf_hist_woba
FROM (SELECT DISTINCT season FROM pa) AS t
JOIN pa AS p
  ON p.season BETWEEN t.season - 3 AND t.season - 1
CROSS JOIN LATERAL (SELECT 6 - (t.season - p.season) AS w)
LEFT JOIN park_factors AS f
  ON f.season = t.season
 AND f.park = p.home_team
GROUP BY t.season, p.batter;
