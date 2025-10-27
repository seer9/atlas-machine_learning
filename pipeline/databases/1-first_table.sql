-- create a table 'first_table' in the database
CREATE TABLE IF NOT EXISTS first_table (
    id INT,
    name VARCHAR(256)
);
USE mysql;
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'rootuser';
FLUSH PRIVILEGES;
