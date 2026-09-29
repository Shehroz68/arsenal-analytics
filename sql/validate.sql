-- 1. rows per season (expect 380 each, the current season will be lower if it is still in progress)
SELECT season, COUNT(*) AS n FROM matches GROUP BY season;

-- 2. nulls in key columns
SELECT SUM(date IS NULL) AS null_date, SUM(fthg IS NULL) AS null_fthg,
       SUM(hs IS NULL) AS null_hs, SUM(b365h IS NULL) AS null_b365h
FROM matches;

-- 3. team name consistency
SELECT DISTINCT hometeam FROM matches ORDER BY 1;

-- 4. result column matches the score (expect 0)
SELECT COUNT(*) AS bad_results FROM matches
WHERE (fthg > ftag AND ftr != 'H') OR (fthg < ftag AND ftr != 'A') OR (fthg = ftag AND ftr != 'D');

-- 5. Arsenal appears identically home and away
SELECT 'home' AS side, COUNT(*) FROM matches WHERE hometeam = 'Arsenal'
UNION ALL
SELECT 'away', COUNT(*) FROM matches WHERE awayteam = 'Arsenal';
