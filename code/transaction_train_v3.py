# -*- coding: utf-8 -*-
'''
Created on 2017.12.20
Updated on 2018.03.14

@author: Qiaowei
'''

import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import os
import sys
from functools import reduce
from operator import mul


# define params
time_step = 30
batch_size = 64
rnn_hidden_unit = 128
input_size = 8
output_layer_1_size = 64
output_size = 1
lstm_depth = 3
output_keep_prob = 0.7
input_keep_prob = 1.0

# learning_rate = 0.0005  # 0.01
init_learning_rate = 0.001
learning_rate_decay = 0.99
init_epoch = 20
max_epoch = 200

l2_regularization_rate = 0.001
print_step = 100
is_training = 1
is_multi_layer = 1
is_predict_by_timestep = 0
continue_training = 0


# model source
model_dir = 'ckpt_198d_8f_321'
dict_dir = model_dir + '_dict'
print('Using model from %s.' % model_dir)
print('Saving dict in path %s.' % dict_dir)

'''
data source
'''
# log_dir = '/qw/data/log_0827_0306_repair/'
training_data_dir = '/qw/data/res_198days_8feature_0827_0312_fixed.csv'
dict_path = '/qw/data/' + dict_dir
print('Training data source is %s.' % training_data_dir)


# import train data
f = open(training_data_dir)
df = pd.read_csv(f)

# 将标准化后的count写入到文件中,以备预测时恢复用
count_mean = np.mean(df['count'])
count_std = np.std(df['count'])
if os.path.isdir(dict_path) is False:
    os.mkdir(dict_path)
normalized_count_data_dir = dict_path + '/normalized_count.dict'
with open(normalized_count_data_dir, 'w') as f:
    f.writelines(str(count_mean)+'\n')
    f.writelines(str(count_std)+'\n')

train_data = df.values

# 将标准化的特征写入到文件中,以备预测时恢复用
normalized_features_mean_data_dir = dict_path + '/normalized_features_mean.npy'
normalized_features_std_data_dir = dict_path + '/normalized_features_std.npy'
np.save(normalized_features_mean_data_dir, np.mean(train_data, axis=0))
np.save(normalized_features_std_data_dir, np.std(train_data, axis=0))

# 标准化训练集
normalized_train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)

# 获取训练集
train_x, train_y = [], []
for i in range(len(normalized_train_data) - time_step - 1):
   x = normalized_train_data[i: i + time_step]
   y = normalized_train_data[i + time_step, 1, np.newaxis]
   train_x.append(x.tolist())
   train_y.append(y.tolist())

# 输入层、输出层权重、偏置
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

learning_rate_to_use = [
    init_learning_rate * (learning_rate_decay**max(float(i + 1 - init_epoch), 0.0)) for i in range(max_epoch)
]
# define LSTM
X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
Y = tf.placeholder(tf.float32, shape=[None, output_size])
learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')

'''
model
'''
def lstm(batch):
    # reshape from 3D to 2D
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, weights['in']) + biases['in']
    # reshape from 2D to 3D
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_hidden_unit])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_unit, state_is_tuple=True)
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

    last_cell_output = output_rnn[:, -1, :]

    # output layer
    rnn_output = tf.reshape(last_cell_output, [-1, rnn_hidden_unit])
    dnn_layer_1 = tf.matmul(rnn_output, weights['out_layer_1']) + biases['out_layer_1']
    pred = tf.matmul(dnn_layer_1, weights['out_layer_2']) + biases['out_layer_2']

    return pred, final_states

'''
get model total params number
'''
def get_num_params():
    num_params = 0
    for variables in tf.trainable_variables():
        shape = variables.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

'''
train LSTM
'''
def train_lstm():
    pred, _ = lstm(batch_size)
    # l2 reg: get all variables
    tv = tf.trainable_variables()
    regularization_cost = l2_regularization_rate * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
    # loss
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1]))) + regularization_cost
    # tf.scalar_summary('loss', loss)
    # params count
    num_params = get_num_params()
    print('Model params total number is %d' %(num_params))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    if continue_training == 1:
        module_file = tf.train.latest_checkpoint('./' + model_dir + '/')  # restore

    with tf.Session() as sess:
        # writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        if continue_training == 1:
            saver.restore(sess, module_file)  # restore

        min_loss = sys.float_info.max
        for epoch_step in range(max_epoch):
            current_learning_rate = learning_rate_to_use[epoch_step]
            step = 0
            start = 0
            end = start + batch_size
            while end < len(train_x):
                _, loss_ = sess.run([train_op, loss], 
                    feed_dict={
                        X: train_x[start:end], 
                        Y: train_y[start:end],
                        learning_rate: current_learning_rate
                    }
                )
                start += batch_size
                end = start + batch_size

                if step % print_step == 0 or loss_ < min_loss:
                    print("Epoch: %d, Step: %d, Loss: %f" %(epoch_step, step, loss_))
                    print("Saving model: ", saver.save(sess, './' + model_dir + '/tp_model.ckpt'))
                    min_loss = loss_
                step += 1


'''
Main
'''
def main(_):
    train_lstm()
    exit()


if __name__ == '__main__':
    tf.app.run()