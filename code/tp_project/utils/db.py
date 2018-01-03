# -*- coding: utf-8 -*-
import pymysql.cursors

class DB(object):

#	__instance = None

	conn = None
	cursor = None

	def __init__(self, host, port, user, passwd, database):
		self.conn = pymysql.Connect(
			host = host,
			port = port,
			user = user,
			passwd = passwd,
			db = database,
			charset = 'utf8',
			cursorclass = pymysql.cursors.DictCursor
		)
		self.cursor = self.conn.cursor()

	def __del__(self):
		self.conn.close()

	def select_one(self, statement):
		self.cursor.execute(statement)
		return self.cursor.fetchone()

	def select(self, statement):
		self.cursor.execute(statement)
		return self.cursor.fetchall()

	def update(self, statement):
		self.cursor.execute(statement)
		self.conn.commit()
	
	def insert(self, statement):
		self.cursor.execute(statement)
		self.conn.commit()
		#print(self.cursor)
