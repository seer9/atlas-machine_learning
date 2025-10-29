-- creates a stored procedure AddBonus
-- adds a new correction for the student
DELIMITER //
CREATE PROCEDURE AddBonus(
    IN student_email VARCHAR(255),
    IN bonus_points INT
)
BEGIN
    UPDATE students
    SET points = points + bonus_points
    WHERE email = student_email;
END //
DELIMITER ;

