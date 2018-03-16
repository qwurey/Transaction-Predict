# -*- coding: utf-8 -*-
'''
Created on 2017年11月28日

@author: Qiao Wei
'''

import datetime
import numpy as np
import time
import urllib3
import json

import holiday
# import special_day

'''
时间转换成时间戳
'''
def datetime2timestamp(dt):
    s = time.mktime(time.strptime(dt,'%Y-%m-%d %H:%M:%S'))
    return int(s)


'''
得到昨天最后30min的数据
'''
def get_data(dt, system, hourfrom='23:30:00', hourto='23:59:00'):
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
        body=encode_data,
        headers={'Content-Type':'application/json', 'apikey':'b2e31497ee69493c88f21a7173ac9724'}
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

'''
根据时间从本地取数据
'''
def get_data_by_localfile(date, system, data_dir, hourfrom='23:30:00', hourto='23:59:00'):
    filename_temp = data_dir + date + '.log'
    index_from = time2index(hourfrom)
    index_to = time2index(hourto)
    res = []
    try:    
        with open(filename_temp, 'r') as f:
            lines_temp = f.readlines()
        dtemp = np.array(lines_temp)
        for i in range(index_from, index_to + 1):
            s = str.split(dtemp[i], '\t')
            d = {}
            d['date'] = s[0]
            d['time'] = s[1]
            d['count'] = float(s[2])
            res.append(d)
    except:
        print('Except in get_data_by_localfile')
        exit()
    return res

'''
23:30:0 -> index
'''
def time2index(time):
    s = str.split(time, ':')
    index = int(s[0]) * 60 + int(s[1])
    return index

'''
index -> 00:00:0
'''
def index2timefrom_to(index):
    datenow = datetime.datetime.now()
    date = datenow.strftime('%Y-%m-%d')
    s = str.split(date, '-')
    # 一天的开始, eg: 2018-01-01 00:00:0
    d0 = datetime.datetime(int(s[0]), int(s[1]), int(s[2]), 0, 0, 0)
    timeto = d0 + datetime.timedelta(minutes=index)  # 增加index分钟
    timefrom = timeto + datetime.timedelta(minutes=0-30) # 减去30min

    s_timefrom = str.split((str)(timefrom), ' ')  # 2018-01-01 00:00:0
    s_timeto = str.split((str)(timeto), ' ')
    return s_timefrom[1], s_timeto[1]


'''
According to the date, return weekday group(5 days) or weekend group(2 days)
'''
def get_weekday_and_weekend_group(date):
    weekday_group = []
    weekend_group = []
    for eachnum in range(1, 8):
        datetmp = (datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=0-eachnum))
        if datetmp.weekday() < 5:
            weekday_group.append(datetmp.strftime('%Y-%m-%d'))
        else:
            weekend_group.append(datetmp.strftime('%Y-%m-%d'))
    return weekday_group, weekend_group

'''
Return last week's same day's data
'''
def get_last_day_data_by_localfile(date, data_dir):
    last_day = np.zeros([1440, 1])
    weekday_group, weekend_group = get_weekday_and_weekend_group(date)

    if isWeekday(date) == 1:
        # read 1 earliest weekday
        filename_temp = data_dir + weekday_group[-1] + '.log'
        with open(filename_temp, 'r') as f:
            lines_temp = f.readlines()
        dtemp = np.array(lines_temp)
        for n in range(len(dtemp)):
            stemp = str.split(dtemp[n], '\t')
            last_day[n] = float(stemp[2])
    else:
        # read 1 earliest weekend
        filename_temp = data_dir + weekend_group[-1] + '.log'
        with open(filename_temp, 'r') as f:
            lines_temp = f.readlines()
        dtemp = np.array(lines_temp)
        for n in range(len(dtemp)):
            stemp = str.split(dtemp[n], '\t')
            last_day[n] = float(stemp[2])
    return last_day


'''
function: judge a date is a holiday or not

return 1: is a holiday
return 0: is not a holiday
'''
def isHoliday(date):
    # date = '2017-10-01'
    if date in holiday.hlist:
        return 1
    else:
        return 0

'''
function: judge a date is a special day or not

return 1: is a special day
return 0: is not a special day
'''
def isSpecialDay(date):
    # date = '2017-10-01'
    if date in special_day.slist:
        return 1
    else:
        return 0

'''
24feature
one-hot encoding: judge time belong to which hour
'''
def handle24(tran, time):
    # time: 00:00:0
    s = str.split(time, ':')
    # is24_0，is24_1，is24_2，is24_3，is24_4，is24_5，is24_6，is24_7，is24_8，is24_9，is24_10，is24_11，is24_12，is24_13，is24_14，is24_15，is24_16，is24_17，is24_18，is24_19，is24_20，is24_21，is24_22，is24_23
    is24_0=0; is24_1=0; is24_2=0; is24_3=0; is24_4=0;
    is24_5=0; is24_6=0; is24_7=0; is24_8=0; is24_9=0;
    is24_10=0; is24_11=0; is24_12=0; is24_13=0; is24_14=0;
    is24_15=0; is24_16=0; is24_17=0; is24_18=0; is24_19=0;
    is24_20=0; is24_21=0; is24_22=0; is24_23=0;
    if s[0] == '00':
        is24_0 = 1
    elif s[0] == '01':
        is24_1 = 1
    elif s[0] == '02':
        is24_2 = 1
    elif s[0] == '03':
        is24_3 = 1
    elif s[0] == '04':
        is24_4 = 1
    elif s[0] == '05':
        is24_5 = 1
    elif s[0] == '06':
        is24_6 = 1
    elif s[0] == '07':
        is24_7 = 1
    elif s[0] == '08':
        is24_8 = 1
    elif s[0] == '09':
        is24_9 = 1
    elif s[0] == '10':
        is24_10 = 1
    elif s[0] == '11':
        is24_11 = 1
    elif s[0] == '12':
        is24_12 = 1
    elif s[0] == '13':
        is24_13 = 1
    elif s[0] == '14':
        is24_14 = 1
    elif s[0] == '15':
        is24_15 = 1
    elif s[0] == '16':
        is24_16 = 1
    elif s[0] == '17':
        is24_17 = 1
    elif s[0] == '18':
        is24_18 = 1
    elif s[0] == '19':
        is24_19 = 1
    elif s[0] == '20':
        is24_20 = 1
    elif s[0] == '21':
        is24_21 = 1
    elif s[0] == '22':
        is24_22 = 1
    elif s[0] == '23':
        is24_23 = 1
    tran.append(is24_0)
    tran.append(is24_1)
    tran.append(is24_2)
    tran.append(is24_3)
    tran.append(is24_4)
    tran.append(is24_5)
    tran.append(is24_6)
    tran.append(is24_7)
    tran.append(is24_8)
    tran.append(is24_9)
    tran.append(is24_10)
    tran.append(is24_11)
    tran.append(is24_12)
    tran.append(is24_13)
    tran.append(is24_14)
    tran.append(is24_15)
    tran.append(is24_16)
    tran.append(is24_17)
    tran.append(is24_18)
    tran.append(is24_19)
    tran.append(is24_20)
    tran.append(is24_21)
    tran.append(is24_22)
    tran.append(is24_23)
    return


'''
12feature
one-hot encoding: judge time belong to which 2hour
'''
def handle12(tran, time):
    # time: 00:00:0
    s = str.split(time, ':')
    is12_0=0; is12_2=0; is12_4=0; is12_6=0; is12_8=0;
    is12_10=0; is12_12=0; is12_14=0; is12_16=0; is12_18=0; is12_20=0; is12_22=0;
    if s[0] == '00' or s[0] == '01':
        is12_0 = 1
    elif s[0] == '02' or s[0] == '03':
        is12_2 = 1
    elif s[0] == '04' or s[0] == '05':
        is12_4 = 1
    elif s[0] == '06' or s[0] == '07':
        is12_6 = 1
    elif s[0] == '08' or s[0] == '09':
        is12_8 = 1
    elif s[0] == '10' or s[0] == '11':
        is12_10 = 1
    elif s[0] == '12' or s[0] == '13':
        is12_12 = 1
    elif s[0] == '14' or s[0] == '15':
        is12_14 = 1
    elif s[0] == '16' or s[0] == '17':
        is12_16 = 1
    elif s[0] == '18' or s[0] == '19':
        is12_18 = 1
    elif s[0] == '20' or s[0] == '21':
        is12_20 = 1
    elif s[0] == '22' or s[0] == '23':
        is12_22 = 1
    tran.append(is12_0)
    tran.append(is12_2)
    tran.append(is12_4)
    tran.append(is12_6)
    tran.append(is12_8)
    tran.append(is12_10)
    tran.append(is12_12)
    tran.append(is12_14)
    tran.append(is12_16)
    tran.append(is12_18)
    tran.append(is12_20)
    tran.append(is12_22)
    return


'''
8feature
one-hot encoding: judge time belong to which 3hour
'''
def handle8(tran, time):
    # time: 00:00:0
    s = str.split(time, ':')
    # is8_0，is8_3，is8_6，is8_9，is8_12，is8_15，is8_18，is8_21
    is8_0=0; is8_3=0; is8_6=0; is8_9=0;
    is8_12=0; is8_15=0; is8_18=0; is8_21=0;
    if s[0] == '00' or s[0] == '01' or s[0] == '02':
        is8_0 = 1
    elif s[0] == '03' or s[0] == '04' or s[0] == '05':
        is8_3 = 1
    elif s[0] == '06' or s[0] == '07' or s[0] == '08':
        is8_6 = 1
    elif s[0] == '09' or s[0] == '10' or s[0] == '11':
        is8_9 = 1
    elif s[0] == '12' or s[0] == '13' or s[0] == '14':
        is8_12 = 1
    elif s[0] == '15' or s[0] == '16' or s[0] == '17':
        is8_15 = 1
    elif s[0] == '18' or s[0] == '19' or s[0] == '20':
        is8_18 = 1
    elif s[0] == '21' or s[0] == '22' or s[0] == '23':
        is8_21 = 1
    tran.append(is8_0)
    tran.append(is8_3)
    tran.append(is8_6)
    tran.append(is8_9)
    tran.append(is8_12)
    tran.append(is8_15)
    tran.append(is8_18)
    tran.append(is8_21)
    return


'''
6feature
one-hot encoding: judge time belong to which 4hour
'''
def handle6(tran, time):
    # time: 00:00:0
    s = str.split(time, ':')
    is6_0=0; is6_4=0; is6_8=0;
    is6_12=0; is6_16=0; is6_20=0;
    if s[0] == '00' or s[0] == '01' or s[0] == '02' or s[0] == '03':
        is6_0 = 1
    elif s[0] == '04' or s[0] == '05'or s[0] == '06' or s[0] == '07':
        is6_4 = 1
    elif s[0] == '08' or s[0] == '09' or s[0] == '10' or s[0] == '11':
        is6_8 = 1
    elif s[0] == '12' or s[0] == '13' or s[0] == '14' or s[0] == '15':
        is6_12 = 1
    elif s[0] == '16' or s[0] == '17' or s[0] == '18' or s[0] == '19':
        is6_16 = 1
    elif s[0] == '20' or s[0] == '21' or s[0] == '22' or s[0] == '23':
        is6_20 = 1
    tran.append(is6_0)
    tran.append(is6_4)
    tran.append(is6_8)
    tran.append(is6_12)
    tran.append(is6_16)
    tran.append(is6_20)
    return


'''
one-hot encoding: judge time belong to which 6hour
'''
def handle4(tran, time):
    # time: 00:00:0
    s = str.split(time, ':')
    # is4_0，is4_6，is4_12，is4_18
    is4_0=0; is4_6=0; is4_12=0; is4_18=0
    if s[0] == '00' or s[0] == '01' or s[0] == '02' or s[0] == '03' or s[0] == '04' or s[0] == '05':
        is4_0 = 1
    elif s[0] == '06' or s[0] == '07' or s[0] == '08' or s[0] == '09' or s[0] == '10' or s[0] == '11':
        is4_6 = 1
    elif s[0] == '12' or s[0] == '13' or s[0] == '14' or s[0] == '15' or s[0] == '16' or s[0] == '17':
        is4_12 = 1
    elif s[0] == '18' or s[0] == '19' or s[0] == '20' or s[0] == '21' or s[0] == '22' or s[0] == '23':
        is4_18 = 1
    tran.append(is4_0)
    tran.append(is4_6)
    tran.append(is4_12)
    tran.append(is4_18)
    return


'''
1feature: current time is which hour
'''
def appendHourDayIndex(tran, time):
    # time: 00:00:0
    s = str.split(time, ':')
    tran.append(int(s[0]))
    return

'''
1feature: current date is weekday or not
'''
def isWeekday(date):
    currentDate = datetime.datetime.strptime(date, '%Y-%m-%d')
    dayOfWeek = currentDate.weekday()
    if dayOfWeek < 5:
        return 1
    else:
        return 0

'''
1feature: current date's weekday index
'''
def appendWeekDayIndex(tran, date):
    currentDate = datetime.datetime.strptime(date, '%Y-%m-%d')
    dayOfWeek = currentDate.weekday()
    tran.append(dayOfWeek)
    return

'''
7feature
one-hot encoding: current date's weekday index
'''
def appendWeekDay(tran, date):
    currentDate = datetime.datetime.strptime(date, '%Y-%m-%d')
    dayOfWeek = currentDate.weekday()
    # isMon, isTue, isWed, isThu, isFri, isSat, isSun
    isMon=0;isTue=0;isWed=0;isThu=0;isFri=0;isSat=0;isSun=0
    if dayOfWeek == 0:
        isMon = 1
    elif dayOfWeek == 1:
        isTue = 1
    elif dayOfWeek == 2:
        isWed = 1
    elif dayOfWeek == 3:
        isThu = 1
    elif dayOfWeek == 4:
        isFri = 1
    elif dayOfWeek == 5:
        isSat = 1
    elif dayOfWeek == 6:
        isSun = 1
    tran.append(isMon)
    tran.append(isTue)
    tran.append(isWed)
    tran.append(isThu)
    tran.append(isFri)
    tran.append(isSat)
    tran.append(isSun)
    return


'''
Return last week's same day's data
'''
def get_last_day_data_by_request(date):
    last_day = np.zeros([1440, 1])
    weekday_group, weekend_group = get_weekday_and_weekend_group(date)

    if isWeekday(date) == 1:
        data_weekday = get_data(weekday_group[-1], 'BOCOP-*', hourfrom='00:00:00', hourto='23:59:00')
    else:
        data_weekday = get_data(weekend_group[-1], 'BOCOP-*', hourfrom='00:00:00', hourto='23:59:00')
    for n in range(len(data_weekday)):
        last_day[n] = float(data_weekday[n]['count'])
    last_day[1439] = float(data_weekday[1438]['count'])
    return last_day


'''
计算相应index的访问量的平均值：取前5天或者前2天的历史数据计算
'''
def get_sum_data_by_request(date):
    sum_data = np.zeros([1440, 1])
    # 取date之前的工作日或周末日的日期存到weekdaygroup or weekendgroup中
    weekday_group, weekend_group = get_weekday_and_weekend_group(date)
    
    if isWeekday(date) == 1:
        # get 5 weekday
        for m in range(len(weekday_group)):
            data_weekday = get_data(weekday_group[m], 'BOCOP-*', hourfrom='00:00:00', hourto='23:59:00')
            for n in range(len(data_weekday) - 1):
                sum_data[n] += float(data_weekday[n]['count'])
            sum_data[1439] += float(data_weekday[1438]['count'])
    else:
        # get 2 weekend
        for m in range(len(weekend_group)):
            data_weekend = get_data(weekend_group[m], 'BOCOP-*', hourfrom='00:00:00', hourto='23:59:00')
            for n in range(len(data_weekend) - 1):
                sum_data[n] += float(data_weekend[n]['count'])
            sum_data[1439] += float(data_weekend[1438]['count'])
    return sum_data

'''
If given date is weekday, return last 5 weekday's mean data
If given date is weekend, return last 2 weekend's mean data
'''
def get_sum_data_by_localfile(date, data_dir):
    sum_data = np.zeros([1440, 1])
    weekday_group, weekend_group = get_weekday_and_weekend_group(date)
    try:
        if isWeekday(date) == 1:
            # read 5 weekday
            for m in range(len(weekday_group)):
                filename_temp = data_dir + weekday_group[m] + '.log'
                with open(filename_temp, 'r') as f:
                    lines_temp = f.readlines()
                dtemp = np.array(lines_temp)
                for n in range(len(dtemp)):
                    stemp = str.split(dtemp[n], '\t')
                    sum_data[n] += float(stemp[2])
        else:
            # read 2 weekend
            for m in range(len(weekend_group)):
                filename_temp = data_dir + weekend_group[m] + '.log'
                with open(filename_temp, 'r') as f:
                    lines_temp = f.readlines()
                dtemp = np.array(lines_temp)
                for n in range(len(dtemp)):
                    stemp = str.split(dtemp[n], '\t')
                    sum_data[n] += float(stemp[2])
    except:
        print('Except in get_sum_data')
        print(weekday_group)
        print(weekend_group)
        print(date)
        exit()
    return sum_data