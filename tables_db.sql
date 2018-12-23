drop database if exists Network_storage;
CREATE DATABASE Network_storage CHARACTER SET utf8 COLLATE utf8_general_ci;
use Network_storage;

-- CREATE ROLE 'app_developer', 'app_read', 'app_write';
--
-- GRANT ALL ON app_db.* TO 'app_developer';
-- GRANT SELECT ON app_db.* TO 'app_read';
-- GRANT INSERT, UPDATE, DELETE ON app_db.* TO 'app_write';
--
-- CREATE USER 'dev1'@'localhost' IDENTIFIED BY 'dev1pass';
-- CREATE USER 'read_user1'@'localhost' IDENTIFIED BY 'read_user1pass';
-- CREATE USER 'read_user2'@'localhost' IDENTIFIED BY 'read_user2pass';
-- CREATE USER 'rw_user1'@'localhost' IDENTIFIED BY 'rw_user1pass';
--
-- GRANT 'app_developer' TO 'dev1'@'localhost';
-- GRANT 'app_read' TO 'read_user1'@'localhost', 'read_user2'@'localhost';
-- GRANT 'app_read', 'app_write' TO 'rw_user1'@'localhost';

CREATE TABLE task_types (
    task_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    task_name VARCHAR(50) NOT NULL
) ENGINE=INNODB;

INSERT INTO `task_types`(`task_name`) VALUES
  ('CLASSIFICATION'),
  ('DETECTION'),
  ('SEGMENTATION');

CREATE TABLE models (
    model_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    task_id INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    FOREIGN KEY (task_id)
        REFERENCES task_types (task_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
) ENGINE=INNODB;

CREATE TABLE networks (
    network_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    model_id INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    info VARCHAR(1000),
    FOREIGN KEY (model_id)
        REFERENCES models (model_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
) ENGINE=INNODB;

CREATE TABLE epochs (
    epoch_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    network_id INT NOT NULL,
    epoch_number INT NOT NULL,
    FOREIGN KEY (network_id)
        REFERENCES networks (network_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
) ENGINE=INNODB;

CREATE TABLE metrics (
    metric_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    metric_name VARCHAR(50) NOT NULL,
    info VARCHAR(1000)
) ENGINE=INNODB;

CREATE TABLE scores (
    score_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    epoch_id INT NOT NULL,
    metric_id INT NOT NULL,
    score_value DECIMAL(7 , 6 ) NOT NULL,
    FOREIGN KEY (epoch_id)
        REFERENCES epochs (epoch_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (metric_id)
        REFERENCES metrics (metric_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
) ENGINE=INNODB;

CREATE TABLE datasets (
    dataset_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    dataset_name VARCHAR(50) NOT NULL,
    dataset_full_path VARCHAR(1000) NOT NULL,
    info VARCHAR(1000)
) ENGINE=INNODB;

CREATE TABLE data_classes (
    data_class_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    dataset_id INT NOT NULL,
    data_class_name VARCHAR(100) NOT NULL,
    FOREIGN KEY (dataset_id)
        REFERENCES datasets (dataset_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
) ENGINE=INNODB;

CREATE TABLE objects (
    object_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    data_class_id INT NOT NULL,
    object_full_path VARCHAR(1000) NOT NULL,
    FOREIGN KEY (data_class_id)
        REFERENCES data_classes (data_class_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
) ENGINE=INNODB;

CREATE TABLE predictions (
    prediction_id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    object_id INT NOT NULL,
    epoch_id INT NOT NULL,
    argmax_value DECIMAL(8 , 5 ) NOT NULL,
    class_index INT NOT NULL,
    FOREIGN KEY (object_id)
        REFERENCES objects (object_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    FOREIGN KEY (epoch_id)
        REFERENCES epochs (epoch_id)
        ON UPDATE CASCADE
        ON DELETE CASCADE
) ENGINE=INNODB;
