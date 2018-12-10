
import tensorflow as tf
import numpy as np

import time
import datetime

a = tf.Variable([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])

b1 = np.array([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])
b2 = np.zeros(shape=[2, 5])

B_mu_in_tensor = tf.placeholder(shape=[2 ,5], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(a - B_mu_in_tensor))
train_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, gate_gradients=0)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    begin = datetime.datetime.now()
    for i in range(20000):
        sess.run(train_optimizer, feed_dict={B_mu_in_tensor: b1})

    print("\n")
    print('当前运行时间：' + str((datetime.datetime.now() - begin)))
    print(sess.run(a))

    for i in range(20000):
        sess.run(train_optimizer, feed_dict={B_mu_in_tensor: b2})

    print("\n")
    print('当前运行时间：' + str((datetime.datetime.now() - begin)))
    print(sess.run(a))

