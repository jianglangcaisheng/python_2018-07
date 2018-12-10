
import tensorflow as tf
import numpy as np
import scipy.io as sio

import time
import datetime

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TYPE = tf.float32

if 0:
    a = tf.Variable([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])
else:
    b_array = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])
    data_step_rate = sio.loadmat(r"F:\0 SoG\0 SOG_201807\0 SOG_201807\matlab_180711\0_data_output/step_1_default.mat")
    data_step_rate = data_step_rate['data_step_rate']
    print("\n")
    print(data_step_rate)

    data_step_rate = np.maximum(data_step_rate, 1e-8)
    print("\nclean 0")
    print(data_step_rate)

    data_step_rate = np.divide(1, data_step_rate)
    print("\n")
    print(data_step_rate)

    data_step_rate = np.sqrt(data_step_rate)
    print("\n")
    print(data_step_rate)

    b_array = data_step_rate
    # b = tf.constant([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]], dtype=TYPE)
    # *2 weak1/4
    b_tensor = tf.placeholder(shape=[3, 24], dtype=TYPE)
    a = tf.multiply(tf.Variable(np.ones(shape=[3, 24]) * b_tensor, dtype=TYPE), 1 / b_tensor)


loss = tf.reduce_sum(tf.square(tf.square(a)))
g = tf.gradients(loss, a)

train_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, gate_gradients=0)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init, feed_dict={b_tensor: b_array})

    begin = datetime.datetime.now()

    train_times = 10
    for i in range(train_times):
        if i % int(train_times / 10) == 0:
            print("数值：")
            print(sess.run(a, feed_dict={b_tensor: b_array}))
            print("\n导数：")
            print(sess.run(g, feed_dict={b_tensor: b_array}))

        sess.run(train_optimizer, feed_dict={b_tensor: b_array})

    print('\n当前运行时间：' + str((datetime.datetime.now() - begin)))



