import sqlite3
import pandas

from ._DataBase import DataBase

# 简单工厂模式，自己的理解：根据传入的参数去选择生成实际的类
# 父类 = 子类 的表现形式


class DataBaseSqlite3(DataBase):
    def __init__(self, database_name, sheet_name):
        # 创建与数据库的连接
        self.__database = sqlite3.connect(database_name)
        # 在内存中创建数据库
        # conn = sqlite3.connect(':memory:')
        self.__sheet_name = sheet_name
        df_data_read = pandas.read_sql(sql="SELECT * FROM " + self.__sheet_name + " LIMIT 1", con=self.__database)
        self.__columns = df_data_read.columns
        self.__id = self.__columns[0]

    def insert(self, document):
        # document是一条字典，应该也可以看成一条json
        try:
            names = ""
            masks = ""
            for name in document.keys():
                names += name + ", "
                masks += "?,"
            # 去掉多余的字符
            names = names[:-2]
            masks = masks[:-1]

            # 将一条数据变成一条元组的形式
            values = []
            for value in document.values():
                values.append(value)
            values = tuple(values)

            # 创建游标cursor对象，该对象的.execute()方法可以执行sql命令
            cursor = self.__database.cursor()
            cursor.execute("INSERT INTO " + self.__sheet_name + "(" + names + ") VALUES(" + masks + ")", values)
            # 连接完数据库并不会自动提交，需要手动commit
            self.__database.commit()
            cursor.close()
            return True
        except Exception as e:
            pass
        return False

    def find(self, filter=None, sort=None):
        sql = "SELECT * FROM " + self.__sheet_name
        # 排序
        if sort is not None and sort[0][0] in self.__columns:
            sql += " ORDER BY " + sort[0][0]
            if sort[0][1] == -1:
                sql += " DESC"
            else:
                sql += " ASC"
        # 利用pandas库里的read_sql
        # pandas.read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None,
        #                 chunksize=None)[source]
        return pandas.read_sql(sql=sql, con=self.__database)

    def find_one(self, filter=None, sort=None):
        sql = "SELECT * FROM " + self.__sheet_name
        if sort is not None and sort[0][0] in self.__columns:
            sql += " ORDER BY " + sort[0][0]
            if sort[0][1] == -1:
                sql += " DESC"
            else:
                sql += " ASC"

        sql += " LIMIT 1"

        return pandas.read_sql(sql=sql, con=self.__database).iloc[0]

    def update_one(self, filter, update):
        print("WARNING: DataBaseSqlite3 update_one does not complete.\n")
        pass

    def delete(self, filter):
        print("WARNING: DataBaseSqlite3 delete does not complete.\n")
        pass

    def close_connect(self):
        self.__database.close()
