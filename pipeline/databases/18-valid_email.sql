-- create a trigger that resets the attribute 'valid_email'
CREATE TRIGGER set_valid_email
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF NEW.email <> OLD.email THEN
        SET new.valid_email = 0;
    END IF;
END;
