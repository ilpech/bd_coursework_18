CREATE USER 'root'@'localhost' IDENTIFIED BY '123root';
CREATE USER 'guest'@'localhost' IDENTIFIED BY '123guest';
CREATE USER 'network'@'localhost' IDENTIFIED BY '123network';

GRANT ALL ON Network_storage.* TO 'root'@'localhost' ;
GRANT SELECT ON Network_storage.* TO 'guest'@'localhost';
GRANT INSERT, UPDATE, SELECT ON Network_storage.* TO 'network'@'localhost';
