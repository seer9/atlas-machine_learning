-- create a trigger that resets the attribute 'valid_email'
CREATE TRIGGER set_valid_email
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF STRCMP(old.email, new.email) THEN
        SET NEW.valid_email = 0;
    END IF;
END;
