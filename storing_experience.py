import sqlite3 as db
from sqlite3 import Error
import re


class StoringExperience:
    def __init__(self):
        self.db_name = "database/experience.db"
        self.create1 = """
            CREATE TABLE IF NOT EXISTS ranking (
            id integer PRIMARY KEY,
            val_acc real,
            val_loss real)
        """
        self.create2 = """
            CREATE TABLE IF NOT EXISTS experience (
            id integer PRIMARY KEY,
            evidence text)
        """

    def connection(self):
        conn = None
        try:
            conn = db.connect(self.db_name)
            return conn
        except Error as e:
            print(e)
        return conn

    def create_db(self):
        conn = self.connection()
        c = conn.cursor()
        try:
            c.execute(self.create1)
            c.execute(self.create2)
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def insert_ranking(self, val_acc, val_loss):
        conn = self.connection()
        c = conn.cursor()
        try:
            c.execute('INSERT INTO ranking (val_acc, val_loss) VALUES (' + str(val_acc) + ',' + str(val_loss) + ')')
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def insert_evidence(self, evidence):
        conn = self.connection()
        c = conn.cursor()
        try:
            c.execute('INSERT INTO experience (evidence) VALUES ("' + str(evidence) + '")')
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def formatting(self, res):
        acc = []
        loss = []
        for i in res:
            acc.append(i[1])
            loss.append(i[2])
        return acc, loss

    def get(self):
        conn = self.connection()
        c = conn.cursor()
        c.execute("SELECT * FROM ranking")
        res = self.formatting(c.fetchall())
        conn.close()
        return res


if __name__ == '__main__':
    se = StoringExperience()
    se.create_db()
    se.insert_ranking(0.7225, 0.7423)
    se.insert_ranking(0.7325, 0.7113)
    se.insert_ranking(0.7895, 0.4353)
    acc, loss = se.get()

    print(acc)
    print()
    print(loss)
