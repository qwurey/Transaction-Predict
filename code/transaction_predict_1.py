# -*- coding: utf-8 -*-
'''
Created on 2017年2月19日

@author: Qiaowei
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# parameters
time_step = 60
rnn_unit = 10
batch_size = 64
input_size = 1
output_size = 1
lr = 0.001
KEEP_PROB = 0.5
is_training = 0
l2_regularization_rate = 0.001

# import training data: 2017-11-13.log_2017-11-23.log: 11*1440 = 15840
f = open('/Users/urey/Projects/GitProject/Transaction-Predict/data/res_11days_1feature.csv')
df = pd.read_csv(f)
# get count col
data = np.array(df['count'])
# normalize training data
normalize_data = (data-np.mean(data))/np.std(data)
# add new axis
normalize_data = normalize_data[:, np.newaxis]


# import test data: 2017-11-24.log
ftest = open('/Users/urey/Projects/GitProject/Transaction-Predict/data/res_log_20171124.csv')
dftest = pd.read_csv(ftest)
datatest = np.array(dftest['count'])
# normalize test data
normalize_data_test = (datatest-np.mean(datatest))/np.std(datatest)
# add new axis
normalize_data_test = normalize_data_test[:, np.newaxis]

# show data
# plt.figure()
# plt.plot(data)
# plt.show()


# create training data
train_x, train_y = [], []
for i in range(len(normalize_data)-time_step-1):
    x = normalize_data[i:i+time_step]
    y = normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

# define RNN
X = tf.placeholder(tf.float32, [None, time_step, input_size])    # 每批次输入网络的tensor
Y = tf.placeholder(tf.float32, [None, time_step, output_size])   # 每批次tensor对应的标签
# input layer & output layer
weights = {
         'in': tf.Variable(tf.random_normal([input_size, rnn_unit], -1.0, 1.0), name='in_w'),
         'out': tf.Variable(tf.random_normal([rnn_unit, 1], -1.0, 1.0), name='out_w')
         }
biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]), name='in_bias'),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]), name='out_bias')
        }

def lstm(batch):
    input = tf.reshape(X, [-1, input_size])  # reshape from 3D to 2D
    input_rnn = tf.matmul(input, weights['in']) + biases['in']
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # reshape from 2D to 3D
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    if is_training:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=KEEP_PROB)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    pred = tf.matmul(output, weights['out']) + biases['out']
    return pred, final_states

# train model
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
    module_file = tf.train.latest_checkpoint(checkpoint_dir='./ckpt60_template/')

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, module_file)

        for i in range(10000):
            step = 0
            start = 0
            end = start + batch_size
            while(end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start + batch_size

                if step%100 == 0:
                    print(i, step, loss_)
                    print("保存模型：", saver.save(sess, './ckpt60_template/stock_model.ckpt'))
                step += 1

# predict
def prediction():
    pred, _ = lstm(1)      # 预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint(checkpoint_dir='./ckpt60/')
        saver.restore(sess, module_file)

        # 取训练集最后一行为测试样本, shape=[1,time_step,input_size]
        prev_seq = train_x[-1]
        predict = []
        # 得到之后1440个预测结果
        for i in range(1440):
            # 每一小时用真实值替换上一小时的预测值,以此类推来预测下一个小时
            if (i != 0) and (i % 60 == 0):
                prev_seq = normalize_data_test[i: i+60]
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

    # 真实值
    real = []
    for i in range(1440):
        real.append(normalize_data_test[i])

    # result
    plt.figure()
    plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
    plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
    plt.plot(list(range(len(normalize_data), len(normalize_data) + len(real))), real, color='y')
    plt.show()

prediction()
