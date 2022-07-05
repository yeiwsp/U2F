from first_web.DATABASE import database_sqlite3


class DataBaseFactory:
    def __init__(self, database_name, sheet_name, model):
        if model == 'sqlite3':
            self.__DataBase = database_sqlite3.DataBaseSqlite3(database_name, sheet_name)

    def get(self):
        return self.__DataBase


