import mysql.connector

from mysql.connector import Error

class Database:
  def __init__(self, host, database, user, pwd):
    self.host = host
    self.database = database
    self.user = user
    self.pwd = pwd
    
  def Connect(self):
    self.connection = None
    try:
      self.connection = mysql.connector.connect(host=self.host, database=self.database, user=self.user, password=self.pwd)
      if self.connection.is_connected():
        print('Connected to MySQL database')
        return True
    except Error as e:
        print(e)
        return False
  
  def Disconnect(self):
    self.connection.close()
    self.connection = None

  def InsertRecord(self, data):
    cursor = self.connection.cursor()
    sql = "INSERT INTO articles (title, content, source, published, analyzed, model1, model2, model3) VALUES("
    for i, key in enumerate(data):
      delimiter = "," if i < len(data) - 1 else ""
      if key == "published" and data[key] == None:
        sql += "{}{}".format("NULL", delimiter)
      elif key is not "model1" and key is not "model2" and key is not "model3":
        sql += "'{}'{}".format(data[key], delimiter)
      else:
        sql += "{}{}".format(data[key], delimiter)
    sql += ");"

    cursor.execute(sql)
    self.connection.commit()
  
  def GetRecords(self, search):
    if search == None:
      sql = "SELECT * from articles ORDER BY id DESC;"
    else:
      sql = "SELECT * from articles WHERE (title LIKE '%{}%' or content LIKE '%{}%' or source LIKE '%{}%') ORDER BY id DESC".format(search, search, search)  
    
    cursor = self.connection.cursor()
    cursor.execute(sql)
    records = cursor.fetchall()
      
    return records


if __name__ == '__main__':
  db = Database("localhost", "sasw", "mjenigar", "SaswDB123!")
  db.Connect()
  db.GetData()