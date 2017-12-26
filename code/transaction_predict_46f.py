# -*- coding: utf-8 -*-
'''
Created on 2017年12月20日

@author: Qiaowei
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

# define params
time_step = 30
batch_size = 64
rnn_unit = 80
input_size = 48
output_size = 1
lstm_depth = 3
output_keep_prob = 0.5
input_keep_prob = 1.0
lr = 0.0005  # 0.01
l2_regularization_rate = 0.001
is_training = 0
is_multi_layer = 1
is_predict_by_timestep = 1

'''
data source
'''
log_dir = '/Users/urey/data/boc/transaction_predict_model/log_1113_1224/'

# 预测周五数据源
# deng_predict_date = '2017-12-15'
# training_data_dir = '/Users/urey/data/boc/transaction_predict_model/res_32days_48feature_1113_1214.csv'
# test_data_dir = '/Users/urey/data/boc/transaction_predict_model/res_1days_48feature_1215.csv'

# 预测周六数据源
deng_predict_date = '2017-12-16'
training_data_dir = '/Users/urey/data/boc/transaction_predict_model/res_33days_48feature_1113_1215.csv'
test_data_dir = '/Users/urey/data/boc/transaction_predict_model/res_1days_48feature_1216.csv'
print('Training data source is %s.' % training_data_dir)
print('Test data source is %s.' % test_data_dir)

# model source
model_dir = 'ckpt_33d_48f_1'  # 测试周五
# model_dir = 'ckpt_33d_48f_1'  # 测试周六
print('Using model from %s.' % model_dir)

# import train data
f = open(training_data_dir)
df = pd.read_csv(f)
# # norm count col
count_mean = np.mean(df['count'])
count_std = np.std(df['count'])
# df['count'] = (df['count'] - count_mean) / count_mean
# # norm index col
# index_mean = np.mean(df['index'])
# index_std = np.std([df['index']])
# df['index'] = (df['index'] - index_mean) / index_std

train_data = df.values
# 标准化训练集
normalized_train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
last_for_predict_x = normalized_train_data[len(normalized_train_data) - time_step: len(normalized_train_data)].tolist()
# 获取训练集
train_x, train_y = [], []
for i in range(len(normalized_train_data) - time_step - 1):
   x = normalized_train_data[i: i + time_step]
   y = normalized_train_data[i + 1: i + time_step + 1, 1, np.newaxis]
   train_x.append(x.tolist())
   train_y.append(y.tolist())

# test data
f_test = open(test_data_dir)
df_test = pd.read_csv(f_test)
# # norm count col
# count_mean_test = np.mean(df_test['count'])
# count_std_test = np.std(df_test['count'])
# df_test['count'] = (df_test['count'] - count_mean_test) / count_std_test
# # norm index col
# index_mean_test = np.mean(df_test['index'])
# index_std_test = np.std([df_test['index']])
# df_test['index'] = (df_test['index'] - index_mean_test) / index_std_test

test_data = df_test.values
# 标准化测试集
normalized_test_data = (test_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)

# 输入层、输出层权重、偏置
weights = {
         'in': tf.Variable(tf.random_normal([input_size, rnn_unit], -1.0, 1.0), name='in_w'),
         'out': tf.Variable(tf.random_normal([rnn_unit, 1], -1.0, 1.0), name='out_w')
        }
biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]), name='in_bias'),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]), name='out_bias')
       }
# define LSTM
X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])

def lstm(batch):
    # reshape from 3D to 2D
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, weights['in']) + biases['in']
    # reshape from 2D to 3D
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    if is_training:
        print('Training lstm nn...')
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)

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

    # 作为输出层的输入
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    pred = tf.matmul(output, weights['out']) + biases['out']
    return pred, final_states

# train LSTM
def train_lstm():
    global batch_size
    pred, _ = lstm(batch_size)
    # l2 reg: get all variables
    tv = tf.trainable_variables()
    regularization_cost = l2_regularization_rate * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
    # loss
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1]))) + regularization_cost
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    # module_file = tf.train.latest_checkpoint('./' + model_dir + '/')  # restore

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)  # restore

        # 重复训练10000次
        for i in range(10000):
            step = 0
            start = 0
            end = start + batch_size
            while end < len(train_x):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start + batch_size

                if step % 100 == 0:
                    print(i, step, loss_)
                    print("保存模型: ", saver.save(sess, './' + model_dir + '/tp_model.ckpt'))
                step += 1

def weighted_average_method(deng_predict_date):
    weekdaygroup = []
    weekendgroup = []

    print('Using weighted average\'s method which is mean to predict %s count.' % deng_predict_date)
    dateFrom = datetime.datetime.strptime(deng_predict_date, '%Y-%m-%d')
    for eachnum in range(1, 8):
        datePredict = datetime.datetime.strptime(deng_predict_date, '%Y-%m-%d')

        datetmp = (dateFrom + datetime.timedelta(days=0 - eachnum))
        if datetmp.weekday() < 5:
            weekdaygroup.append(datetmp.strftime('%Y-%m-%d'))
        else:
            weekendgroup.append(datetmp.strftime('%Y-%m-%d'))
    dayOfWeek = datePredict.weekday()
    if dayOfWeek < 5:
        print('It\'s a weekday, so only use 5 history day.')
        res = weighted_average_algorithm(weekdaygroup)
    else:
        print('It\'s a weekend, so only use 2 history day.')
        res = weighted_average_algorithm(weekendgroup)
    res = [(elem - count_mean) / count_std for elem in res]
    print('weighted average\'s method is over.')
    return res

def weighted_average_algorithm(dategroup):
    tmpdata = []
    predata = []
    for i in range(0, 1440):
        predata.append(0)
        tmpdata.append(0)

    j = 5
    for dt in dategroup:
        filename = log_dir + dt + '.log'
        print(filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
        data = np.array(lines)
        for i in range(len(data)):
            ds = str.split(data[i], '\t')
            tmpdata[i] = (int)(ds[2])
            predata[i] += tmpdata[i] * j / 15
        j = j - 1
    return predata

def prediction():
    pred, _ = lstm(1)      # 预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint(checkpoint_dir='./' + model_dir + '/')
        saver.restore(sess, module_file)
        # 取训练集最后一行为测试样本, prev_seq shape=[1,time_step,input_size]
        prev_seq = last_for_predict_x
        predict = []
        # 得到之后1440个预测结果
        if is_predict_by_timestep:
            print('Predicting by timestep...')
        for i in range(1440):
            # 每一小时用真实值替换上一小时的预测值,以此类推来预测下一个小时
            if is_predict_by_timestep and (i != 0) and (i % time_step == 0):
                prev_seq = normalized_test_data[i - time_step: i]
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            # 每次得到最后一个时间步的预测结果，先构造一个Xt的数据
            test_sample = normalized_test_data[i].tolist()
            test_sample[1] = next_seq[-1]
            # 再与之前的数据加在一起，形成新的测试样本
            prev_seq = np.vstack((prev_seq[1:], test_sample))

    # 真实值
    real = []
    for i in range(1440):
        real.append(normalized_test_data[i, 1])

    res_wa = weighted_average_method(deng_predict_date)

    sum = 0
    for i in range(1440):
        sum += (real[i] - predict[i]) * (real[i] - predict[i])
    accuracy_rate_lstm = sum/1440
    print('Evaluating, accuracy_rate_lstm = %d', accuracy_rate_lstm)
    # evaluate: real, res_wa
    sum = 0
    for i in range(1440):
        sum += (real[i] - res_wa[i]) * (real[i] - res_wa[i])
    accuracy_rate_deng = sum/1440
    print('Evaluating, accuracy_rate_deng = %d', accuracy_rate_deng)

    # result
    plt.figure()
    l1, = plt.plot(list(range(len(normalized_train_data))), normalized_train_data[:, 1], color='b')
    l2, = plt.plot(list(range(len(normalized_train_data), len(normalized_train_data) + len(res_wa))), res_wa, color='g')
    l3, = plt.plot(list(range(len(normalized_train_data), len(normalized_train_data) + len(predict))), predict, color='r')
    l4, = plt.plot(list(range(len(normalized_train_data), len(normalized_train_data) + len(real))), real, color='y')
    plt.legend(handles=[l1, l2, l3, l4], labels=['training data', 'mean: weighted average\'s method', 'lstm', 'real data'], loc='best')
    plt.show()

# train_lstm()
# exit()
prediction()
