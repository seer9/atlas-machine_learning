-- procedure 'ComputeAverageScoreForUser' that calculates the average score for each student
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    DECLARE avg_score FLOAT;

    SELECT AVG(score) INTO avg_score
    FROM corrections
    WHERE user_id = user_id;

    SELECT avg_score AS average_score;
END //
DELIMITER ;