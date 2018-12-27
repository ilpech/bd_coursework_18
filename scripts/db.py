import datetime
import sys
import time
import mysql.connector
from mysql.connector import errorcode
from pypika import Table, Query, Field

class NumpyMySQLConv(mysql.connector.conversion.MySQLConverter):
    def _float32_to_mysql(self, value):
        return float(value)

    def _float64_to_mysql(self, value):
        return float(value)

    def _int32_to_mysql(self, value):
        return int(value)

    def _int64_to_mysql(self, value):
        return int(value)

class DB:
    def __init__(self, **kwargs):
        self.login = kwargs['login']
        self.password = kwargs['password']
        self.config = self.db_access(self.login, self.password)
        self.cnx = mysql.connector.connect(**self.config)
        self.cnx.set_converter_class(NumpyMySQLConv)
        self.cur = self.cnx.cursor()

    def db_access(self, login, password):
        config = {
            'host': 'localhost',
            'port': 3306,
            'database': 'Data_analysis_dev',
            'user': login,
            'password': password,
            'charset': 'utf8',
            'use_unicode': True,
            'get_warnings': True,
        }
        return config

    def create_table(self, TABLES):
        for name, ddl in TABLES.items():
            try:
                print("Creating table {}: ".format(name), end='')
                self.cur.execute(ddl)
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("already exists.")
                else:
                    raise err
            else:
                print("Table {} was successfully created".format(name))

    def repair_query(self, q):
        q = str(q).replace('"', '`')
        return q

    def select(self, query):
        try:
            self.cur.execute(self.repair_query(query), params=None, multi=False)
            data = self.cur.fetchall()
            return data
        except mysql.connector.Error as err:
            raise err

    def insert(self, query):
        try:
            self.cur.execute(self.repair_query(query))
            self.cnx.commit()
        except mysql.connector.Error as err:
            raise err

    def update(self, query):
        try:
            self.cur.execute(self.repair_query(query))
            self.cnx.commit()
        except mysql.connector.Error as err:
            raise err


def get_db_access():
    return DB(login='root', password=getpass.getpass('DataBase root password: '))
