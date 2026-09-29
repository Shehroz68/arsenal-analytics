-- Step 2: modelling layer on top of the raw `matches` table.

-- One row per team per match (long format). Every downstream feature is built from this.
DROP VIEW IF EXISTS team_matches;
CREATE VIEW team_matches AS
SELECT match_id, season, date, hometeam AS team, awayteam AS opponent, 1 AS is_home,
       fthg AS gf, ftag AS ga, hs AS shots_f, "as" AS shots_a, hst AS sot_f, ast AS sot_a,
       hc AS corners_f, ac AS corners_a, hy AS yellow, hr AS red,
       CASE ftr WHEN 'H' THEN 3 WHEN 'D' THEN 1 ELSE 0 END AS points
FROM matches
UNION ALL
SELECT match_id, season, date, awayteam, hometeam, 0,
       ftag, fthg, "as", hs, ast, hst, ac, hc, ay, ar,
       CASE ftr WHEN 'A' THEN 3 WHEN 'D' THEN 1 ELSE 0 END
FROM matches;

-- Final league table per season.
DROP VIEW IF EXISTS season_table;
CREATE VIEW season_table AS
SELECT season, team,
       COUNT(*) AS played, SUM(points = 3) AS won, SUM(points = 1) AS drawn, SUM(points = 0) AS lost,
       SUM(gf) AS gf, SUM(ga) AS ga, SUM(gf) - SUM(ga) AS gd, SUM(points) AS pts,
       RANK() OVER (PARTITION BY season ORDER BY SUM(points) DESC, SUM(gf) - SUM(ga) DESC, SUM(gf) DESC) AS position
FROM team_matches
GROUP BY season, team;

-- Arsenal season summary, including shot-quality proxies.
DROP VIEW IF EXISTS arsenal_seasons;
CREATE VIEW arsenal_seasons AS
SELECT t.season, t.position, t.pts, t.gf, t.ga,
       ROUND(1.0 * SUM(tm.sot_f) / NULLIF(SUM(tm.shots_f), 0), 3) AS sot_share,
       ROUND(1.0 * SUM(tm.gf) / NULLIF(SUM(tm.sot_f), 0), 3) AS goals_per_sot,
       ROUND(1.0 * SUM(tm.ga) / NULLIF(SUM(tm.sot_a), 0), 3) AS conceded_per_sot_against
FROM season_table t
JOIN team_matches tm ON tm.season = t.season AND tm.team = t.team
WHERE t.team = 'Arsenal'
GROUP BY t.season;
