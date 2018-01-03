from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from utils.hbase.hbase import THBaseService
from utils.hbase.hbase.ttypes import *

HBASE_HOST = '192.168.210.30'
HBASE_PORT = 9090

class HBase(object):

	__instance = None
	
	transport = None
	client = None

	def __init__(self):
		transport = TSocket.TSocket(HBASE_HOST, HBASE_PORT)
		self.transport = TTransport.TBufferedTransport(transport)
		protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
		self.client = THBaseService.Client(protocol)
		self.transport.open()
		print('Connect to hbase success')

	def __new__(cls, *args, **kwds):
		if not cls.__instance:
			cls.__instance = super(HBase, cls).__new__(cls, *args, **kwds)
		return cls.__instance

	def __del__(self):
		self.transport.close()
		print('Closed connection from hbase')

	def __formatTResult(self, tResult):
		d = {}
		for tColumnValue in tResult.columnValues:
			family = tColumnValue.family.decode('utf-8')
			qualifier = tColumnValue.qualifier.decode('utf-8')
			value = tColumnValue.value.decode('utf-8')
			if family not in d:
				d[family] = {}
			d[family][qualifier] = value
		res = {
			'row_key': tResult.row.decode('utf-8'),
			'columnValues': d 
		}
		return d

	def readRow(self, table, rowKey):
		tGet = TGet()
		tGet.row = rowKey.encode(encoding='utf-8')
		tResult = self.client.get(table.encode(encoding='utf-8'), tGet)
		return self.__formatTResult(tResult)

	def readCell(self, table, rowKey, family, qualifier):
		tColumn = TColumn()
		tColumn.family = family.encode(encoding='utf-8')
		tColumn.qualifier = qualifier.encode(encoding='utf-8')
		tGet = TGet()
		tGet.row = rowKey.encode(encoding='utf-8')
		tGet.columns = [tColumn]
		tResult = self.client.get(table.encode(encoding='utf-8'), tGet)
		return tResult.columnValues[0].value.decode('utf-8')

	def writeCell(self, table, rowKey, family, qualifier, value):
		tColumnValue = TColumnValue()
		tColumnValue.family = family.encode(encoding='utf-8')
		tColumnValue.qualifier = qualifier.encode(encoding='utf-8')
		tColumnValue.value = value.encode(encoding='utf-8')
		tPut = TPut()
		tPut.row = rowKey.encode(encoding='utf-8')
		tPut.columnValues = [tColumnValue]
		self.client.put(table.encode(encoding='utf-8'), tPut)

	def deleteCell(self, table, rowKey, family, qualifier):
		tColumn = TColumn()
		tColumn.family = family.encode(encoding='utf-8')
		tColumn.qualifier = qualifier.encode(encoding='utf-8')
		tDelete = TDelete()
		tDelete.row = rowKey.encode(encoding='utf-8')
		tDelete.columns = [tColumn]
		self.client.deleteSingle(table.encode(encoding='utf-8'), tDelete)

	def scan(self, table, startRow, families, rowNum):
		columns = []
		for i in range(len(families)):
			tColumn = TColumn()
			tColumn.family = families[i].encode(encoding='utf-8')
			columns.append(tColumn)
		tScan = TScan()
		tScan.startRow = startRow.encode(encoding='utf-8')
		tScan.columns = columns
		tResults = self.client.getScannerResults(table.encode(encoding='utf-8'), tScan, rowNum)
		res = []
		for i in range(len(tResults)):
			res.append(self.__formatTResult(tResults[i]))
		return res
