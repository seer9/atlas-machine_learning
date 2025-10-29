-- create a trigger to decrease the quanity of an item column after a new order
CREATE TRIGGER decrease_quantity
AFTER INSERT ON orders
FOR EACH ROW
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE name = NEW.item_name;
