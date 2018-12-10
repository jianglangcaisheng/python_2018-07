
import tensorflow as tf
import numpy as np

import time
import datetime

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TYPE = tf.float32

if 0:
    a = tf.Variable([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])
else:
    b_array = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])
    # b = tf.constant([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]], dtype=TYPE)
    # *2 weak1/4
    b_tensor = tf.placeholder(shape=[2, 5], dtype=TYPE)
    a = tf.multiply(tf.Variable(np.ones(shape=[2, 5]) * b_tensor, dtype=TYPE), 1 / b_tensor)


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



