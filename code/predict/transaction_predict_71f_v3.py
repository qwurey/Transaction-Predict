# -*- coding: utf-8 -*-
'''
Created on 2017年12月20日

@author: Qiaowei
'''

import tensorflow as tf
import numpy as np
import datetime
from utils.db import DB
import handle_features
import params
import func

'''
LSTM config
'''
time_step = 60
batch_size = 64
rnn_hidden_unit = 128
input_size = 8
output_layer_1_size = 80
output_size = 1
lstm_depth = 3
output_keep_prob = 0.5
input_keep_prob = 1.0

l2_regularization_rate = 0.001
is_training = 0
is_multi_layer = 1
is_predict_by_timestep = 0

'''
DB config
'''
system = 'BOCOP-*'
db_eops = DB(host='192.168.130.30', port=3306, user='root', passwd='Password01!', database='eops')

'''
model source
'''
model_dir = 'ckpt_169d_8f_306'
print('Using model from %s.' % model_dir)
print('Using dict from %s' %(params.project_dir + '/' + params.dict_dir))
# 恢复标准化的交易量
normalized_count_data_dir = params.project_dir + '/' + params.dict_dir + '/normalized_count.dict'
with open(normalized_count_data_dir, 'r') as f:
    data = f.readlines()
    count_mean = float(data[0])
    count_std = float(data[1])

'''
LSTM defination
'''
weights = {
         'in': tf.Variable(tf.random_normal([input_size, rnn_hidden_unit], -1.0, 1.0), name='in_w'),
         'out_layer_1': tf.Variable(tf.random_normal([rnn_hidden_unit, output_layer_1_size], -1.0, 1.0), name='out_layer_1_w'),
         'out_layer_2': tf.Variable(tf.random_normal([output_layer_1_size, output_size], -1.0, 1.0), name='out_layer_2_w')
        }
biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_hidden_unit, ]), name='in_bias'),
        'out_layer_1': tf.Variable(tf.constant(0.1, shape=[output_layer_1_size, ]), name='out_layer_1_bias'),
        'out_layer_2': tf.Variable(tf.constant(0.1, shape=[output_size, ]), name='out_layer_2_bias')
       }

X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
Y = tf.placeholder(tf.float32, shape=[None, output_size])

def lstm(batch):
    # reshape from 3D to 2D
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, weights['in']) + biases['in']
    # reshape from 2D to 3D
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_hidden_unit])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_unit, state_is_tuple=True)
    if is_training:
        print('Training lstm nn. Using dropout layer.')
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
    else:
        print('Predicting. Not using dropout layer.')

    if is_multi_layer:
        # multi lstm layer
        print('Using multi lstm layer.')
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * lstm_depth, state_is_tuple=True)
        init_state = stacked_lstm.zero_state(batch, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(stacked_lstm, input_rnn, initial_state=init_state, dtype=tf.float32, time_major=False)
    else:
        # single lstm layer
        print('Using single lstm layer.')
        init_state = lstm_cell.zero_state(batch, dtype=tf.float32)
        # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
        output_rnn, final_states = tf.nn.dynamic_rnn(lstm_cell, input_rnn, initial_state=init_state, dtype=tf.float32, time_major=False)

    last_cell_output = output_rnn[:, -1, :]

    # output layer
    rnn_output = tf.reshape(last_cell_output, [-1, rnn_hidden_unit])
    dnn_layer_1 = tf.matmul(rnn_output, weights['out_layer_1']) + biases['out_layer_1']
    pred = tf.matmul(dnn_layer_1, weights['out_layer_2']) + biases['out_layer_2']

    return pred, final_states
    

'''
predict next 1440min series
'''
def prediction():
    pred, _ = lstm(1)      # 预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint(checkpoint_dir='./' + model_dir + '/')
        saver.restore(sess, module_file)
        # 取上一天的最后30分钟的数据为测试样本, prev_seq shape=[1,time_step,input_size]
        prev_seq = last_for_predict_x
        predict = []
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        sum_data = None
        last_weekday_data = None
        # sum_data = func.get_sum_data_by_request(today)
        # last_weekday_data = func.get_last_day_data_by_request(today)
        sum_data = func.get_sum_data_by_localfile(today, data_dir)
        last_weekday_data = func.get_last_day_data_by_localfile(today, data_dir)
        # print(sum_data)
        # print(last_weekday_data)
        # exit()
        # 得到之后1440个预测结果
        if is_predict_by_timestep:
            print('Predicting by timestep...')
        for i in range(1440):
            # 每经过time_step分钟，用真实值替换上一个time_step所有分钟的预测值,以此类推来预测下一个time_step的所有分钟值
            # if is_predict_by_timestep and (i != 0) and (i % time_step == 0):
            #     timefrom, timeto = func.index2timefrom_to(i)
            #     # print("timefrom=%s, timeto=%s" %(timefrom, timeto))
            #     real_data = handle_features.get_data(today, system, timefrom, timeto)
            #     real_data = real_data[0: time_step]
            #     real_vector = handle_features.create_6features_vector(real_data, today)
            #     # print(real_vector)
            #     normalized_real_vector = handle_features.normalized_feature_vector(real_vector)
            #     # print(len(normalized_real_vector))
            #     prev_seq = normalized_real_vector
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            # print(next_seq)
            predict_count = np.float32(next_seq[-1]).item()
            # print(predict_count)
            predict.append(predict_count)
            # 每次得到最后一个时间步的预测结果，先构造一个Xt的特征向量下一分钟
            sample_vector = handle_features.create_next_min_8feature_vector(i, prev_seq, predict_count, count_mean, count_std, sum_data, last_weekday_data)
            # print(sample_vector)
            normalized_sample_vector = handle_features.normalized_feature_vector(sample_vector)
            # print('sv:')
            # print(sample_vector)
            # print('nsv；')
            # print(normalized_sample_vector)
            # print(prev_seq)
            
            # 再与之前的数据加在一起，形成新的测试样本
            prev_seq = np.vstack((prev_seq[1:], normalized_sample_vector))
            # print(prev_seq)
            # exit()
    # exit()
    print('Insert to DB...')
    # 生成下一天的1440个时间刻度
    datenow = datetime.datetime.now().strftime('%Y-%m-%d')
    datenow_tmp = datetime.datetime.strptime(datenow, '%Y-%m-%d')
    dategroup = []
    for i in range(0, 1440):
        dategroup.append((datenow_tmp+datetime.timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S'))

    # 取消标准化,生成交易量预测值
    for i in range(0, 1440):
        predict[i] = predict[i] * count_std + count_mean

    # 将交易量基线插入到db中
    sys = system
    # database-test: bocop_minute_web_qwlstm
    # database-online: bocop_minute_web_pre
    for i in range(0, 1440):
        line = "INSERT INTO `bocop_minute_web_pre` (`SYS_DATE`,`SYSTEM`,`SYS_TIME`,`LOW_WEB_NUM`,`UP_WEB_NUM`) VALUES ('%s', '%s', '%s', %.2f, %.2f)" % (datenow, sys, dategroup[i], predict[i]*0.8, predict[i]*1.2)
        db_eops.insert(line)
    print('Insert to DB succeed!')
    return


'''
每天的00:03调用该方法预测当天的1440个分钟的交易量
'''

'''
拿到引导预测数据
'''
# get yesterday's date
data_dir = '/qw/data_temp/'
datenow = datetime.datetime.now()
yesterday = (datenow - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

'''
time_step = 30 -> '23:30:00', '23:59:00'
time_step = 60 -> '23:00:00', '23:59:00'
'''
# get yesterday's last 30 minutes' data
# last_30min_data = func.get_data(yesterday, system, '23:30:00', '23:59:00')
last_30min_data = func.get_data_by_localfile(yesterday, system, data_dir, '23:00:00', '23:59:00')
# last_30min_data = func.get_data_by_localfile(yesterday, system, data_dir, '22:00:00', '23:59:00')

'''
by_request
last_30min_data[time_step - 1]['count'] = last_30min_data[time_step - 2]['count']
'''

sum_data = None
last_weekday_data = None
sum_data = func.get_sum_data_by_localfile(yesterday, data_dir)
last_weekday_data = func.get_last_day_data_by_localfile(yesterday, data_dir)

# transfer data to feature vector
sample_features_vector = handle_features.create_features_vector(last_30min_data, yesterday, sum_data, last_weekday_data)
# print(sample_features_vector)
# exit()
# norm
last_for_predict_x = handle_features.normalized_feature_vector(sample_features_vector)
# print(last_for_predict_x)
# exit()

print('Start to predict...')
prediction()
print('Predict succeed! Please check result on web.')
