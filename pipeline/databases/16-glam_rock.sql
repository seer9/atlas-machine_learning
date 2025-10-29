-- ranks country origins of bands, ordered by the number of fans
-- ranks country origins of glam rock bands, ordered by the number of fans
SELECT origin, SUM(fans) nb_fans
FROM metal_bands
WHERE genre = 'Glam Rock'
GROUP BY origin
ORDER BY nb_fans DESC;