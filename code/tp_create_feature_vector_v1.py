# -*- coding: utf-8 -*-
'''
Created on 2017.11.28
Modified on 2018.02.22

@author: Qiao Wei

transaction predict model: create feature vector on the original data set
'''

import pandas as pd
import numpy as np
import datetime
import func

'''
data source config
'''
date = '2017-08-27'
dateFrom = datetime.datetime.strptime(date, '%Y-%m-%d')
dayLong = 192
dateTo = (dateFrom + datetime.timedelta(days=dayLong-1)).strftime('%Y-%m-%d')
data_dir = '/qw/data/log_0827_0306_repair/'

# saveFile = '/qw/data/res_141days_7feature_0827_0114_fixed.csv'
saveFile = '/qw/data/res_192days_8feature_0827_0306_fixed.csv'
features_num = 8


'''
only handle most important features
'''
def process8f(filename, date, sum_data, last_day):
    # read target day's total logs: 2017-09-01.log
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = np.array(lines)

    global res
    for i in range(len(data)):
        # data[i]: 2017-08-27(ds[0])  00:13:0(ds[1])  3997(ds[2])    0(ds[3])
        ds = str.split(data[i], '\t')

        tran = []

        #1 index: 当前分钟索引
        tran.append(i)

        #2 count: 交易量
        tran.append(float(ds[2]))

        #3 isWeekday: 工作日
        if func.isWeekday(ds[0]) == 1:
            tran.append(1)
        else:
            tran.append(0)

        #4 每天的第几个小时
        func.appendHourDayIndex(tran, ds[1])

        #5 每周的第几天
        func.appendWeekDayIndex(tran, ds[0])

        #6 计算相应index分钟的5天或者2天的访问量的平均值
        mean_index = 0.0
        if sum_data is not None:
            if func.isWeekday(date) == 1:  # 前5个工作日的相应index分钟的平均访问量
                mean_index = sum_data[i] / 5.0
            else:  # 前2个周末日的相应index分钟的平均访问量
                mean_index = sum_data[i] / 2.0
            tran.append(float(mean_index))
        else:
            tran.append(float(ds[2]))

        #7 上一周该天相应index分钟的访问量
        if last_day is not None:
            tran.append(float(last_day[i]))
        else:
            tran.append(float(ds[2]))

        #8 是否是节假日
        holiday = func.isHoliday(ds[0])
        tran.append(holiday)

        res.append(tran)
    return


'''
only handle most important features
'''
def process7f(filename, date, sum_data, last_day):
    # read target day's total logs: 2017-09-01.log
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = np.array(lines)

    global res
    for i in range(len(data)):
        # data[i]: 2017-08-27(ds[0])  00:13:0(ds[1])  3997(ds[2])    0(ds[3])
        ds = str.split(data[i], '\t')

        tran = []

        # index: 当前分钟索引
        tran.append(i)

        # count: 交易量
        tran.append(float(ds[2]))

        # isWeekday: 工作日
        if func.isWeekday(ds[0]) == 1:
            tran.append(1)
        else:
            tran.append(0)

        # 每天的第几个小时
        func.appendHourDayIndex(tran, ds[1])

        # 每周的第几天
        func.appendWeekDayIndex(tran, ds[0])

        # 计算相应index分钟的5天或者2天的访问量的平均值
        mean_index = 0.0
        if sum_data is not None:
            if func.isWeekday(date) == 1:  # 前5个工作日的相应index分钟的平均访问量
                mean_index = sum_data[i] / 5.0
            else:  # 前2个周末日的相应index分钟的平均访问量
                mean_index = sum_data[i] / 2.0
            tran.append(float(mean_index))
        else:
            tran.append(float(ds[2]))

        # 上一周该天相应index分钟的访问量
        if last_day is not None:
            tran.append(float(last_day[i]))
        else:
            tran.append(float(ds[2]))
        res.append(tran)
    return

'''
only handle most important features
'''
def process6f(filename, date, sum_data, last_day):
    # read target day's total logs: 2017-09-01.log
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = np.array(lines)

    global res
    for i in range(len(data)):
        # data[i]: 2017-08-27(ds[0])  00:13:0(ds[1])  3997(ds[2])    0(ds[3])
        ds = str.split(data[i], '\t')

        tran = []

        # index: 当前分钟索引
        tran.append(i)

        # count: 交易量
        tran.append(float(ds[2]))

        # isWeekday: 工作日
        if func.isWeekday(ds[0]) == 1:
            tran.append(1)
        else:
            tran.append(0)

        # 每周的第几天
        func.appendWeekDayIndex(tran, ds[0])

        # 计算相应index分钟的5天或者2天的访问量的平均值
        mean_index = 0.0
        if sum_data is not None:
            if func.isWeekday(date) == 1:  # 前5个工作日的相应index分钟的平均访问量
                mean_index = sum_data[i] / 5.0
            else:  # 前2个周末日的相应index分钟的平均访问量
                mean_index = sum_data[i] / 2.0
            tran.append(float(mean_index))
        else:
            tran.append(float(ds[2]))

        # 上一周该天相应index分钟的访问量
        if last_day is not None:
            tran.append(float(last_day[i]))
        else:
            tran.append(float(ds[2]))
        res.append(tran)
    return


'''
only handle most important features
'''
def process4f(filename, date):
    # read target day's total logs: 2017-09-01.log
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = np.array(lines)

    global res
    for i in range(len(data)):
        # data[i]: 2017-08-27(ds[0])  00:13:0(ds[1])  3997(ds[2])    0(ds[3])
        ds = str.split(data[i], '\t')

        tran = []

        # index: 当前分钟索引
        tran.append(i)

        # count: 交易量
        tran.append(float(ds[2]))

        # isWeekday: 工作日
        if func.isWeekday(ds[0]) == 1:
            tran.append(1)
        else:
            tran.append(0)

        # 每周的第几天
        func.appendWeekDayIndex(tran, ds[0])

        res.append(tran)
    return

'''
only handle most important features
'''
def process5f(filename, date, sum_data):
    # read target day's total logs: 2017-09-01.log
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = np.array(lines)

    global res
    for i in range(len(data)):
        # data[i]: 2017-08-27(ds[0])  00:13:0(ds[1])  3997(ds[2])    0(ds[3])
        ds = str.split(data[i], '\t')

        tran = []

        # index: 当前分钟索引
        tran.append(i)

        # count: 交易量
        tran.append(float(ds[2]))

        # isWeekday: 工作日
        if func.isWeekday(ds[0]) == 1:
            tran.append(1)
        else:
            tran.append(0)

        # weekday_index：每周的第几天
        func.appendWeekDayIndex(tran, ds[0])
        
        # mean_index_count：计算相应index分钟的5天或者2天的访问量的平均值
        mean_index = 0.0
        if sum_data is not None:
            if func.isWeekday(date) == 1:  # 前5个工作日的相应index分钟的平均访问量
                mean_index = sum_data[i] / 5.0
            else:  # 前2个周末日的相应index分钟的平均访问量
                mean_index = sum_data[i] / 2.0
            tran.append(float(mean_index))
        else:
            tran.append(float(ds[2]))

        res.append(tran)
    return

'''
'''
def saveFeatures(date, dayLong, saveFile):
    date_from = datetime.datetime.strptime(date, '%Y-%m-%d')
    
    global res
    for i in range(dayLong):
        target_date = (date_from + datetime.timedelta(days=0+i)).strftime('%Y-%m-%d')
        filename = data_dir + target_date + '.log'
        # print(filename)
        if features_num == 8 or features_num == 9:
            # 增加相应index分钟的平均访问量的特征, 根据今天是否是工作日,拿到前5个工作日的数据或者前2个周末日的数据
            sum_data = None
            # 增加上周该日相应index分钟的访问量特征
            last_day = None
            if i >= 7:
                sum_data = func.get_sum_data_by_localfile(target_date, data_dir)
                last_day = func.get_last_day_data_by_localfile(target_date, data_dir)

            process8f(filename, target_date, sum_data, last_day)
        elif features_num == 7 or features_num == 6:
            # 增加相应index分钟的平均访问量的特征, 根据今天是否是工作日,拿到前5个工作日的数据或者前2个周末日的数据
            sum_data = None
            # 增加上周该日相应index分钟的访问量特征
            last_day = None
            if i >= 7:
                sum_data = func.get_sum_data_by_localfile(target_date, data_dir)
                last_day = func.get_last_day_data_by_localfile(target_date, data_dir)

            process7f(filename, target_date, sum_data, last_day)
        elif features_num == 5:
            sum_data = None
            if i >= 7:
                sum_data = func.get_sum_data_by_localfile(target_date, data_dir)
            process5f(filename, target_date, sum_data)
        elif features_num == 4:
            process4f(filename, target_date)
        else:
            print('Error: process features')
            exit()

    # exit()
    df = pd.DataFrame(res)  # write to the file
    df.to_csv(saveFile, mode='a', index=False, header=headers8)

headers4 = ['index', 'count', 'isWeekday', 'weekday_index']

headers5 = ['index', 'count', 'isWeekday', 'weekday_index', 'mean_index_count']

headers6 = ['index', 'count', 'isWeekday', 'weekday_index', 'mean_index_count', 'last_same_weekday_count']

headers7 = ['index', 'count', 'isWeekday', 'hourofday_index', 'weekday_index', 'mean_index_count',
            'last_same_weekday_count'
            ]

# headers8 = ['index', 'count', 'isWeekday', 'hourofday_index', 'hour2ofday_index', 'weekday_index', 'mean_index_count',
#             'last_same_weekday_count'
#             ]
headers8 = ['index', 'count', 'isWeekday', 'hourofday_index', 'weekday_index', 'mean_index_count',
            'last_same_weekday_count', 'is_holiday'
            ]

headers9 = ['index', 'count', 'is_weekday', 'hourofday_index', 'hour2ofday_index', 'weekday_index', 'mean_index_count',
            'last_same_weekday_count', 'is_holiday'
            ]

headers48 = ['index', 'count', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri', 'isSat',
           'isSun', 'isWeekday', 'is24_0', 'is24_1', 'is24_2', 'is24_3',
           'is24_4', 'is24_5', 'is24_6', 'is24_7', 'is24_8', 'is24_9', 'is24_10',
           'is24_11', 'is24_12', 'is24_13', 'is24_14', 'is24_15', 'is24_16', 'is24_17',
           'is24_18', 'is24_19', 'is24_20', 'is24_21', 'is24_22', 'is24_23', 'is8_0',
           'is8_3', 'is8_6', 'is8_9', 'is8_12', 'is8_15', 'is8_18', 'is8_21', 'is4_0',
           'is4_6', 'is4_12', 'is4_18', 'hourdayIndex', 'weekdayIndex']

headers67 = ['index', 'count', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri', 'isSat',
           'isSun', 'isWeekday', 'is24_0', 'is24_1', 'is24_2', 'is24_3',
           'is24_4', 'is24_5', 'is24_6', 'is24_7', 'is24_8', 'is24_9', 'is24_10',
           'is24_11', 'is24_12', 'is24_13', 'is24_14', 'is24_15', 'is24_16', 'is24_17',
           'is24_18', 'is24_19', 'is24_20', 'is24_21', 'is24_22', 'is24_23', 'is8_0',
           'is8_3', 'is8_6', 'is8_9', 'is8_12', 'is8_15', 'is8_18', 'is8_21', 'is4_0',
           'is4_6', 'is4_12', 'is4_18', 'hourdayIndex', 'weekdayIndex', 'is12_0',
           'is12_2', 'is12_4', 'is12_6', 'is12_8', 'is12_10', 'is12_12', 'is12_14',
           'is12_16', 'is12_18', 'is12_20', 'is12_22', 'is6_0', 'is6_4', 'is6_8',
           'is6_12', 'is6_16', 'is6_20', 'meanIndex'
             ]

headers71 = ['index', 'count', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri', 'isSat',
           'isSun', 'isWeekday', 'is24_0', 'is24_1', 'is24_2', 'is24_3',
           'is24_4', 'is24_5', 'is24_6', 'is24_7', 'is24_8', 'is24_9', 'is24_10',
           'is24_11', 'is24_12', 'is24_13', 'is24_14', 'is24_15', 'is24_16', 'is24_17',
           'is24_18', 'is24_19', 'is24_20', 'is24_21', 'is24_22', 'is24_23', 'is8_0',
           'is8_3', 'is8_6', 'is8_9', 'is8_12', 'is8_15', 'is8_18', 'is8_21', 'is4_0',
           'is4_6', 'is4_12', 'is4_18', 'hourdayIndex', 'weekdayIndex', 'is12_0',
           'is12_2', 'is12_4', 'is12_6', 'is12_8', 'is12_10', 'is12_12', 'is12_14',
           'is12_16', 'is12_18', 'is12_20', 'is12_22', 'is6_0', 'is6_4', 'is6_8',
           'is6_12', 'is6_16', 'is6_20', 'mean_3min', 'mean_5min', 'mean_8min', 'mean_10min',
           'meanIndex',
           ]


res = []
saveFeatures(date, dayLong, saveFile)
print('Day from %s to %s log have all generated features.' % (dateFrom, dateTo))
print('Generate features successfully!')
print('The features file is located in %s' % saveFile)






'''
process every log file, transfer every record to a feature vector: 71 features

@params:
    filename: /Users/urey/data/boc/transaction_predict_model/log_0827_1227_repair/2017-08-27.log
    date: 2017-08-27
    has_mean_index: mean_index feature is real or not
'''
def process(filename, date, has_mean_index):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = np.array(lines)

    if has_mean_index is True:
        weekday_group = []
        weekend_group = []
        for eachnum in range(1, 8):
            datetmp = (datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=0-eachnum))
            if datetmp.weekday() < 5:
                weekday_group.append(datetmp.strftime('%Y-%m-%d'))
            else:
                weekend_group.append(datetmp.strftime('%Y-%m-%d'))
        sum_data = np.zeros([1440, 1])
        if func.isWeekday(date) == 1:
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

    global res
    for i in range(len(data)):
        # data[i]: 2017-08-27      00:13:0 3997    0
        ds = str.split(data[i], '\t')
        tran = []
        # index: 当前分钟索引
        tran.append(i)
        # count: 交易量
        tran.append(float(ds[2]))
        # 星期几: isMon, isTue, isWed, isThu, isFri, isSat, isSun
        # func.appendWeekDay(tran, ds[0])

        # isWeekday: 工作日
        if func.isWeekday(ds[0]) == 1:
            tran.append(1)
        else:
            tran.append(0)

        # 时点24维: is24_0，is24_1，is24_2，is24_3，is24_4，is24_5，is24_6，
        # is24_7，is24_8，is24_9，is24_10，is24_11，is24_12，is24_13，is24_14，
        # is24_15，is24_16，is24_17，is24_18，is24_19，is24_20，is24_21，is24_22，is24_23
        # func.handle24(tran, ds[1])
        # 时点8维:is8_0，is8_3，is8_6，is8_9，is8_12，is8_15，is8_18，is8_21
        func.handle8(tran, ds[1])
        # 时点4维: is4_0，is4_6，is4_12，is4_18
        func.handle4(tran, ds[1])
        # 每天的第几个小时
        func.appendHourDayIndex(tran, ds[1])
        # 每周的第几天
        func.appendWeekDayIndex(tran, ds[0])

        # 时点12维:is12_0，is12_2，is12_4，is12_6，is12_8，is12_10，is12_12，is12_14，is12_16，is12_18，is12_20，is12_22
        func.handle12(tran, ds[1])
        # 时点6维:is6_0，is6_4，is6_8，is6_12，is6_16，is6_20
        func.handle6(tran, ds[1])
        '''
        # mean_3min, mean_5min, mean_8min, mean_10min
        # 得到上一天的最后10个数据
        lastday_10data = []
        # 前3分钟访问量
        if i >= 3:
            mean_3min = 0.0
            for j in range(3):
                temp_data = str.split(data[i - j - 1], '\t')
                mean_3min += float(temp_data[2])
            mean_3min /= 3.0
            tran.append(float(mean_3min))
        else:
            tran.append(float(ds[2]))
        # 前5分钟访问量
        if i >= 5:
            mean_5min = 0.0
            for j in range(5):
                temp_data = str.split(data[i - j - 1], '\t')
                mean_5min += float(temp_data[2])
            mean_5min /= 5.0
            tran.append(float(mean_5min))
        else:
            tran.append(float(ds[2]))
        # 前8分钟访问量
        if i >= 8:
            mean_8min = 0.0
            for j in range(8):
                temp_data = str.split(data[i - j - 1], '\t')
                mean_8min += float(temp_data[2])
            mean_8min /= 8.0
            tran.append(float(mean_8min))
        else:
            tran.append(float(ds[2]))
        # 前10分钟访问量
        if i >= 10:
            mean_10min = 0.0
            for j in range(10):
                temp_data = str.split(data[i - j - 1], '\t')
                mean_10min += float(temp_data[2])
            mean_10min /= 10.0
            tran.append(float(mean_10min))
        else:
            tran.append(float(ds[2]))
        '''
        # 计算相应index的访问量的平均值
        mean_index = 0.0
        if has_mean_index is True:
            if func.isWeekday(date) == 1:  # 前5个工作日的相应index分钟的平均访问量
                mean_index = sum_data[i] / 5.0
            else:  # 前2个周末日的相应index分钟的平均访问量
                mean_index = sum_data[i] / 2.0
            tran.append(float(mean_index))
        else:
            tran.append(float(ds[2]))

        # 1. 每月的第几周: 该特征待加入
        # 2. 每月的第几天: 该特征待加入
        # 3. isHoliday: 节假日: 该特征待加入
        # if func.isHoliday(ds[0]) == 1:
        #     tran.append(1)
        # else:
        #     tran.append(0)
        # 4. isSpecial: 特殊日: 该特征待加入
        # if func.isSpecialDay(ds[0]) == 1:
        #     tran.append(1)
        # else:
        #     tran.append(0)

        res.append(tran)
    return
