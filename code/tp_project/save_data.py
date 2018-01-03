# -*- coding: utf-8 -*-
#from utils.db import DB
import json
import base64
import datetime
import urllib3
import time

def datetime2timestamp(dt):
	s = time.mktime(time.strptime(dt,'%Y-%m-%d %H:%M:%S'))
	return int(s)

def getData(dt, system, hourfrom='00:00:00', hourto='23:59:00'):
	datefrom = datetime2timestamp(dt + ' ' + hourfrom)
	datetill = datetime2timestamp(dt + ' ' + hourto)
	http = urllib3.PoolManager()
	data = {
	"aggs": {
		"date_histogram": {
			"field": "@timestamp",
			"interval": "1m",   
			"time_zone": "Asia/Shanghai",
			"min_doc_count": 1
			}
		},
	"filter": {
		"system": system,
		"software": "Apache"
		},
	"range": {
		"time_from": datefrom,
		"time_till": datetill
		}
	} 
	encode_data = json.dumps(data).encode('utf-8')
	
	r = http.request(
		'POST',
		'http://21.122.16.209/log/agg/logstash*',
		body = encode_data,
		headers = {'Content-Type':'application/json', 'apikey':'b2e31497ee69493c88f21a7173ac9724'}
	)
	
	res = []
	
	res_tmp = []
	res_tmp = json.loads(r.data.decode('utf-8'))['data']
	for each in res_tmp:
		tmp = {}
		hour = int(each['key_as_string'][11:13])
		minute =int(each['key_as_string'][14:16])
		index = 60*hour + minute
		tmp['date'] = each['key_as_string'][0:10]
		tmp['time'] = each['key_as_string'][11:18]
		tmp['isWeekday'] = 1 if datetime.datetime.strptime(dt, '%Y-%m-%d').weekday() < 5 else 0
		tmp['count'] = each['doc_count']
		#print (tmp)
		res.append(tmp)
	return res
	
def saveData1(dt, system="BOCOP-*"):
	filename = "/qw/data_temp/" + str(dt) + ".log"
	with open(filename, 'a') as f:
		res = getData(dt, system)
		for record in res:
			data = record['date'] + '\t' + record['time'] + '\t' + str(record['count']) + '\t' + str(record['isWeekday']) + '\n'
			f.write(data)

def saveData2(dt, system="BOCOP-*"):
	# filename = "/qw/data/" + str(dt) + ".log"
	# with open(filename, 'a') as f:
	res = getData(dt, system)
	for record in res:
		data = record['date'] + '\t' + record['time'] + '\t' + str(record['count']) + '\t' + str(record['isWeekday']) + '\n'
		f.write(data)

def saveDataEveryTwoHours(dt, system="BOCOP-*"):
	filename = "/qw/data_temp/" + str(dt) + ".log"
	timedict = [
			['00:00:00', '05:59:00'],       
			['06:00:00', '11:59:00'],       
			['12:00:00', '17:59:00'],       
			['18:00:00', '23:59:00']
		]
	with open(filename, 'a') as f:
		for i in range(4):
			res = getData(dt, system, timedict[i][0], timedict[i][1])
			for record in res:
				data = record['date'] + '\t' + record['time'] + '\t' + str(record['count']) + '\t' + str(record['isWeekday']) + '\n'
				f.write(data)

datefrom = datetime.datetime.strptime('2017-12-28', '%Y-%m-%d')
dateto = datetime.datetime.strptime('2017-12-29', '%Y-%m-%d')
delta = dateto - datefrom

if delta.days < 0:
	print ('Error, dateto < datefrom, please set correct date number!')
	exit()
# every day has a log file
print ("Ready to download log from (%s) to (%s)..." %(datefrom, dateto))	
for day in range(0, delta.days + 1):
	date = (dateto + datetime.timedelta(days=0-day)).strftime('%Y-%m-%d')
	print ("Ready to download the log on (%s)..." % (date))
	saveDataEveryTwoHours(date)
	print ("The log on (%s) has downloaded successfully!" %(date))
print ("All the data from (%s) to (%s) have downloaded successfully!" %(datefrom, dateto))		
				

# only one log file
# filename = "/qw/data/data.log"
# with open(filename, 'a') as f:
# 	print ("Ready to download log from (%s) to (%s)..." %(datefrom, dateto))	
# 	for day in range(0, delta.days + 1):
# 		date = (dateto + datetime.timedelta(days=0-day)).strftime('%Y-%m-%d')
# 		res = getData(date, system="BOCOP-*")
# 		for record in res:
# 			data = record['date'] + '\t' + record['time'] + '\t' + str(record['count']) + '\t' + str(record['isWeekday']) + '\n'
# 			f.write(data)
# 		print ("The log on (%s) has wrote successfully!" %(date))
# print ("All the data from (%s) to (%s) have downloaded successfully!" %(datefrom, dateto))	
