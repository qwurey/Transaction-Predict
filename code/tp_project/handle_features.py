# -*- coding: utf-8 -*-
'''
Created on 2017年12月20日

@author: Qiaowei
'''

import numpy as np
import datetime
import time
import json
import func
import urllib3
import params

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
计算相应index的访问量的平均值：取前5天或者前2天的历史数据计算
'''
def get_sum_data(date):
    
    # 取date之前的工作日或周末日的日期存到weekdaygroup or weekendgroup中
    weekdaygroup = []
    weekendgroup = []
    for eachnum in range(1, 8):
        datetmp = (datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=0-eachnum))
        if datetmp.weekday() < 5:
            weekdaygroup.append(datetmp.strftime('%Y-%m-%d'))
        else:
            weekendgroup.append(datetmp.strftime('%Y-%m-%d'))
    
    # 发请求去取数据
    sum_data = np.zeros([1440, 1])
    if func.isWeekday(date) == 1:
        # get 5 weekday
        for m in range(len(weekdaygroup)):
            data_weekday = get_data(weekdaygroup[m], 'BOCOP-*', hourfrom='00:00:00', hourto='23:59:00')
            for n in range(len(data_weekday)):
                sum_data[n] += float(data_weekday[n]['count'])
    else:
        # get 2 weekend
        for m in range(len(weekendgroup)):
            data_weekend = get_data(weekendgroup[m], 'BOCOP-*', hourfrom='00:00:00', hourto='23:59:00')
            for n in range(len(data_weekend)):
                sum_data[n] += float(data_weekend[n]['count'])
    return sum_data

'''
构造一个下一分钟Xt的特征向量：71维
'''
def create_next_min_feature_vector(index, prev_seq, predict_count, count_mean, count_std, sum_data):

    datenow = datetime.datetime.now()
    date = datenow.strftime('%Y-%m-%d')
    s = str.split(date, '-')
    # 一天的开始, eg: 2018-01-01 00:00:0
    d0 = datetime.datetime(int(s[0]), int(s[1]), int(s[2]), 0, 0, 0)
    d1 = d0 + datetime.timedelta(minutes=index)  # 增加index分钟

    ss = str.split((str)(d1), ' ')  # 2018-01-01 00:00:0

    # 开始构造
    tran = []
    # index: 当前分钟索引
    tran.append(index)
    # count: 交易量
    tran.append(predict_count * count_std + count_mean)
    # 星期几: isMon, isTue, isWed, isThu, isFri, isSat, isSun
    func.appendWeekDay(tran, date)

    # isWeekday: 工作日
    if func.isWeekday(date) == 1:
        tran.append(1)
    else:
        tran.append(0)

    # 时点24维: is24_0，is24_1，is24_2，is24_3，is24_4，is24_5，is24_6，is24_7，is24_8，is24_9，is24_10，is24_11，is24_12，is24_13，is24_14，is24_15，is24_16，is24_17，is24_18，is24_19，is24_20，is24_21，is24_22，is24_23
    func.handle24(tran, ss[1])
    # 时点8维:is8_0，is8_3，is8_6，is8_9，is8_12，is8_15，is8_18，is8_21
    func.handle8(tran, ss[1])
    # 时点4维: is4_0，is4_6，is4_12，is4_18
    func.handle4(tran, ss[1])
    # 每天的第几个小时
    func.appendHourDayIndex(tran, ss[1])
    # 每周的第几天
    func.appendWeekDayIndex(tran, date)
    
    # 时点12维:is12_0，is12_2，is12_4，is12_6，is12_8，is12_10，is12_12，is12_14，is12_16，is12_18，is12_20，is12_22
    func.handle12(tran, ss[1])
    # 时点6维:is6_0，is6_4，is6_8，is6_12，is6_16，is6_20
    func.handle6(tran, ss[1])

    '''
    # mean_3min, mean_5min, mean_8min, mean_10min
    len_prev_seq = len(prev_seq) # 用来计算mean_3min, mean_5min, mean_8min, mean_10min 4个feature

    # 前3分钟访问量
    mean_3min = 0.0
    for j in range(3):
        mean_3min += (float(prev_seq[len_prev_seq - j - 1][1]) * count_std + count_mean)
    mean_3min /= 3.0
    tran.append(mean_3min)

    # 前5分钟访问量
    mean_5min = 0.0
    for j in range(5):
        mean_5min += (float(prev_seq[len_prev_seq - j - 1][1]) * count_std + count_mean)
    mean_5min /= 5.0
    tran.append(mean_5min)

    # 前8分钟访问量
    mean_8min = 0.0
    for j in range(8):
        mean_8min += (float(prev_seq[len_prev_seq - j - 1][1]) * count_std + count_mean)
    mean_8min /= 8.0
    tran.append(mean_8min)

    # 前10分钟访问量
    mean_10min = 0.0
    for j in range(10):
        mean_10min += (float(prev_seq[len_prev_seq - j - 1][1]) * count_std + count_mean)
    mean_10min /= 10.0
    tran.append(mean_10min)
    '''

    mean_index = 0.0
    if func.isWeekday(date) == 1:  # 前5个工作日的相应index分钟的平均访问量
        mean_index = sum_data[index] / 5.0
    else:  # 前2个周末日的相应index分钟的平均访问量
        mean_index = sum_data[index] / 2.0
    tran.append(mean_index[0])

    return tran

'''
构造一个下一分钟Xt的特征向量：6维
'''
def create_next_min_6feature_vector(index, prev_seq, predict_count, count_mean, count_std, sum_data):

    datenow = datetime.datetime.now()
    date = datenow.strftime('%Y-%m-%d')
    s = str.split(date, '-')
    # 一天的开始, eg: 2018-01-01 00:00:0
    d0 = datetime.datetime(int(s[0]), int(s[1]), int(s[2]), 0, 0, 0)
    d1 = d0 + datetime.timedelta(minutes=index)  # 增加index分钟

    ss = str.split((str)(d1), ' ')  # 2018-01-01 00:00:0

    # 开始构造
    tran = []
    # index: 当前分钟索引
    tran.append(index)
    # count: 交易量
    tran.append(predict_count * count_std + count_mean)

    # isWeekday: 工作日
    if func.isWeekday(date) == 1:
        tran.append(1)
    else:
        tran.append(0)

    # 每天的第几个小时
    func.appendHourDayIndex(tran, ss[1])
    # 每周的第几天
    func.appendWeekDayIndex(tran, date)

    mean_index = 0.0
    if func.isWeekday(date) == 1:  # 前5个工作日的相应index分钟的平均访问量
        mean_index = sum_data[index] / 5.0
    else:  # 前2个周末日的相应index分钟的平均访问量
        mean_index = sum_data[index] / 2.0
    tran.append(mean_index[0])

    return tran

'''
构建特征向量：71维
'''
def create_features_vector(data, date, sum_data):
    res = []

    for i in range(len(data)):
        tran = []
        # index: 当前分钟索引
        index = min2index(data[i]['time'])
        tran.append(index)
        # count: 交易量
        tran.append(data[i]['count'])
        # 星期几: isMon, isTue, isWed, isThu, isFri, isSat, isSun
        func.appendWeekDay(tran, data[i]['date'])

        # isWeekday: 工作日
        if func.isWeekday(data[i]['date']) == 1:
            tran.append(1)
        else:
            tran.append(0)

        # 时点24维: is24_0，is24_1，is24_2，is24_3，is24_4，is24_5，is24_6，
        # is24_7，is24_8，is24_9，is24_10，is24_11，is24_12，is24_13，is24_14，
        # is24_15，is24_16，is24_17，is24_18，is24_19，is24_20，is24_21，is24_22，is24_23
        func.handle24(tran, data[i]['time'])
        # 时点8维:is8_0，is8_3，is8_6，is8_9，is8_12，is8_15，is8_18，is8_21
        func.handle8(tran, data[i]['time'])
        # 时点4维: is4_0，is4_6，is4_12，is4_18
        func.handle4(tran, data[i]['time'])
        # 每天的第几个小时
        func.appendHourDayIndex(tran, data[i]['time'])
        # 每周的第几天
        func.appendWeekDayIndex(tran, data[i]['date'])
        # 时点12维:is12_0，is12_2，is12_4，is12_6，is12_8，is12_10，is12_12，is12_14，is12_16，is12_18，is12_20，is12_22
        func.handle12(tran, data[i]['time'])
        # 时点6维:is6_0，is6_4，is6_8，is6_12，is6_16，is6_20
        func.handle6(tran, data[i]['time'])
        '''
        # 前3分钟访问量
        if i >= 3:
            mean_3min = 0.0
            for j in range(3):
                mean_3min += float(data[i - j - 1]['count'])
            mean_3min /= 3.0
            tran.append(mean_3min)
        else:
            tran.append(float(data[i]['count']))
        # 前5分钟访问量
        if i >= 5:
            mean_5min = 0.0
            for j in range(5):
                mean_5min += float(data[i - j - 1]['count'])
            mean_5min /= 5.0
            tran.append(mean_5min)
        else:
            tran.append(float(data[i]['count']))
        # 前8分钟访问量
        if i >= 8:
            mean_8min = 0.0
            for j in range(8):
                mean_8min += float(data[i - j - 1]['count'])
            mean_8min /= 8.0
            tran.append(mean_8min)
        else:
            tran.append(float(data[i]['count']))
        # 前10分钟访问量
        if i >= 10:
            mean_10min = 0.0
            for j in range(10):
                mean_10min += float(data[i - j - 1]['count'])
            mean_10min /= 10.0
            tran.append(mean_10min)
        else:
            tran.append(float(data[i]['count']))
        '''
        # 计算相应index的访问量的平均值
        mean_index = 0.0
        if func.isWeekday(date) == 1:  # 前5个工作日的相应index分钟的平均访问量
            mean_index = sum_data[index] / 5.0
        else:  # 前2个周末日的相应index分钟的平均访问量
            mean_index = sum_data[index] / 2.0
        tran.append(mean_index[0])


        '''
        # 每月的第几周
        # 每月的第几天
        # isHoliday: 节假日
        # if func.isHoliday(ds[0]) == 1:
        #     tran.append(1)
        # else:
        #     tran.append(0)
        # isSpecial: 特殊日
        # if func.isSpecialDay(ds[0]) == 1:
        #     tran.append(1)
        # else:
        #     tran.append(0)
        '''

        res.append(tran)
    return res

'''
构建特征向量：6维
'''
def create_6features_vector(data, date, sum_data):
    res = []

    for i in range(len(data)):
        tran = []
        # index: 当前分钟索引
        index = min2index(data[i]['time'])
        tran.append(index)
        
        # count: 交易量
        tran.append(data[i]['count'])

        # isWeekday: 工作日
        if func.isWeekday(data[i]['date']) == 1:
            tran.append(1)
        else:
            tran.append(0)

        # 每天的第几个小时
        func.appendHourDayIndex(tran, data[i]['time'])
        
        # 每周的第几天
        func.appendWeekDayIndex(tran, data[i]['date'])

        # 计算相应index的访问量的平均值
        mean_index = 0.0
        if func.isWeekday(date) == 1:  # 前5个工作日的相应index分钟的平均访问量
            mean_index = sum_data[index] / 5.0
        else:  # 前2个周末日的相应index分钟的平均访问量
            mean_index = sum_data[index] / 2.0
        tran.append(mean_index[0])

        res.append(tran)
    return res

'''
标准化给定的特征向量
'''
def normalized_feature_vector(feature_vector):
    # 获取训练数据标准化的mean和std
    normalized_features_mean_data_dir = params.project_dir + '/' + params.dict_dir + '/normalized_features_mean.npy'
    normalized_features_std_data_dir = params.project_dir + '/' + params.dict_dir + '/normalized_features_std.npy'

    train_data_mean = np.load(normalized_features_mean_data_dir)
    train_data_std = np.load(normalized_features_std_data_dir)
    return (feature_vector - train_data_mean) / train_data_std


'''
标准化：待测试去掉是否可以
'''
def normalized_next_min_feature_vector(tran):
    # 获取训练数据标准化的mean和std
    normalized_features_mean_data_dir = params.project_dir + '/' + params.dict_dir + '/normalized_features_mean.npy'
    normalized_features_std_data_dir = params.project_dir +  '/' + params.dict_dir + '/normalized_features_std.npy'

    train_data_mean = np.load(normalized_features_mean_data_dir)
    train_data_std = np.load(normalized_features_std_data_dir)
    return (tran - train_data_mean) / train_data_std

'''
example:
    transfer '00:00:00' to 0
    transfer '00:11:00' to 11
    transfer '23:59:00' to 1439
'''
def min2index(min):
    s = str.split(min, ':')
    index = (int)(s[0]) * 60 + (int)(s[1])
    return index

'''
Example:
tran = [1439, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 23, 2]
'''
