-- creates a stored procedure AddBonus
-- adds a new correction for the student
DELIMITER //
CREATE PROCEDURE AddBonus(
    IN user_id INT, 
    IN project_name VARCHAR(255),
    IN score INT)
BEGIN 
    DECLARE project_id INT;

    SELECT id INTO project_id 
    FROM projects 
    WHERE name = project_name;
    IF project_id IS NOT NULL THEN
        INSERT INTO corrections (user_id, project_id, points) 
        VALUES (user_id, project_id, 1);
    END IF;
END //
DELIMITER ;
