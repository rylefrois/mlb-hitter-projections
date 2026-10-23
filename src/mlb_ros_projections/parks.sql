-- Park factors and each hitter's exposure to them.
-- Needs: pa, snapshot_dates (features.sql), venues (venues.py).
--
-- A park is the actual ballpark (MLB venue id), not the home team, so team
-- moves (Athletics to Sacramento, Rays to Steinbrenner Field in 2025) and
-- neutral-site games (Tokyo, London, Bristol) are handled correctly. Venue
-- ids survive renamings, e.g. Safeco Field -> T-Mobile Park.


-- Each game's ballpark.
CREATE OR REPLACE TABLE game_venue AS
SELECT DISTINCT p.season, p.game_pk, p.game_date, p.home_team, p.away_team, v.venue_id
FROM pa AS p
LEFT JOIN venues AS v USING (game_pk);


-- Totals at each ballpark and host team, both teams combined.
CREATE OR REPLACE TABLE park_season AS
SELECT
    p.season,
    g.venue_id,
    p.home_team,
    count(*)          AS pa,
    sum(is_k::INT)    AS k,
    sum(is_ubb::INT)  AS ubb,
    sum(woba_denom)   AS woba_denom,
    sum(woba_value)   AS woba_num
FROM pa AS p
JOIN game_venue AS g USING (game_pk)
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


-- Park factor = rate at the ballpark / rate in the host team's road games
-- that same season. The same teams play on both sides, so team quality
-- cancels out. Factors for a season use only the previous three seasons
-- (skipping 2020), so they are known at forecast time. 1.10 means the park
-- inflates that stat by 10%.
--
-- Ballparks with under 2000 PA in that window get no factor (treated as
-- neutral): one-off neutral sites, and a new park in its first season.
CREATE OR REPLACE TABLE park_factors AS
WITH target AS (
    SELECT DISTINCT season FROM pa
),
window_rows AS (
    SELECT t.season AS target_season, h.*
    FROM target AS t
    JOIN park_season AS h
      ON h.season BETWEEN t.season - 3 AND t.season - 1
     AND h.season <> 2020
),
home AS (
    SELECT target_season, venue_id,
           sum(pa) AS pa, sum(k) AS k, sum(ubb) AS ubb,
           sum(woba_denom) AS woba_denom, sum(woba_num) AS woba_num
    FROM window_rows
    GROUP BY ALL
),
road AS (
    SELECT w.target_season, w.venue_id,
           sum(r.pa) AS pa, sum(r.k) AS k, sum(r.ubb) AS ubb,
           sum(r.woba_denom) AS woba_denom, sum(r.woba_num) AS woba_num
    FROM window_rows AS w
    JOIN team_road_season AS r
      ON r.season = w.season
     AND r.team = w.home_team
    GROUP BY ALL
),
latest_name AS (
    SELECT venue_id, arg_max(venue_name, season) AS venue_name FROM venues GROUP BY venue_id
)
SELECT
    home.target_season AS season,
    venue_id,
    n.venue_name,
    home.pa AS home_pa,
    (home.k   / home.pa) / (road.k   / road.pa)                         AS k_rate,
    (home.ubb / home.pa) / (road.ubb / road.pa)                         AS ubb_rate,
    (home.woba_num / home.woba_denom) / (road.woba_num / road.woba_denom) AS woba
FROM home
JOIN road USING (target_season, venue_id)
LEFT JOIN latest_name AS n USING (venue_id)
WHERE home.pa >= 2000;


-- Average park factor over the PA each hitter has had so far this season.
-- Ballparks with no factor count as neutral.
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
JOIN game_venue AS g USING (game_pk)
LEFT JOIN park_factors AS f
  ON f.season = p.season
 AND f.venue_id = g.venue_id
GROUP BY s.season, s.asof_date, p.batter;


-- The team each hitter last batted for before each forecast date.
CREATE OR REPLACE TABLE batter_team_at_date AS
SELECT s.season, s.asof_date, p.batter, arg_max(p.bat_team, p.game_date) AS team
FROM snapshot_dates AS s
JOIN pa AS p
  ON p.season = s.season
 AND p.game_date < s.asof_date
GROUP BY s.season, s.asof_date, p.batter;


-- Average park factor over the hitter's team's remaining schedule. The
-- schedule is published before the season, so this is known at forecast time.
CREATE OR REPLACE TABLE park_exposure_ros AS
WITH team_remaining AS (
    SELECT
        s.season,
        s.asof_date,
        t.team,
        avg(coalesce(f.k_rate, 1))   AS pf_ros_k_rate,
        avg(coalesce(f.ubb_rate, 1)) AS pf_ros_ubb_rate,
        avg(coalesce(f.woba, 1))     AS pf_ros_woba
    FROM snapshot_dates AS s
    JOIN (SELECT DISTINCT season, home_team AS team FROM game_venue) AS t
      ON t.season = s.season
    JOIN game_venue AS g
      ON g.season = s.season
     AND g.game_date >= s.asof_date
     AND t.team IN (g.home_team, g.away_team)
    LEFT JOIN park_factors AS f
      ON f.season = g.season
     AND f.venue_id = g.venue_id
    GROUP BY s.season, s.asof_date, t.team
)
SELECT d.season, d.asof_date, d.batter, r.pf_ros_k_rate, r.pf_ros_ubb_rate, r.pf_ros_woba
FROM batter_team_at_date AS d
JOIN team_remaining AS r USING (season, asof_date, team);


-- Park factor behind each hitter's Marcel projection: their PA over the
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
JOIN game_venue AS g USING (game_pk)
CROSS JOIN LATERAL (SELECT 6 - (t.season - p.season) AS w)
LEFT JOIN park_factors AS f
  ON f.season = t.season
 AND f.venue_id = g.venue_id
GROUP BY t.season, p.batter;
