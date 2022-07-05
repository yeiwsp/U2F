# Create your models here.

from first_web.DATABASE import *


class Model:
    def __init__(self, database_name, sheet_name, model):
        self.__database = DataBaseFactory.DataBaseFactory(database_name,
                                                          sheet_name,
                                                          model).get()

    def getDataBase(self):
        return self.__database




