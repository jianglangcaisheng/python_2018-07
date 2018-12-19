
import tensorflow as tf
import numpy as np

import time
import datetime

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TYPE = tf.float32

is_constant = False

a = tf.Variable(0.)
shape = (24000, 100)
if is_constant:
    goal = tf.ones(shape=shape)
else:
    goal = tf.placeholder(shape=shape, dtype=TYPE)
    goal_numpy = np.ones(shape=shape)


loss = tf.square(a - tf.reduce_sum(goal))
# g = tf.gradients(loss, a)

train_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, gate_gradients=0)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    if is_constant:
        sess.run(init)
    else:
        sess.run(init, feed_dict={goal: goal_numpy})

    begin = datetime.datetime.now()

    train_times = 1500
    for i in range(train_times):
        if i % int(train_times / 10) == 0 and 0:
            print("数值：")
            print(sess.run(a, feed_dict={goal: goal_numpy}))
            print("\n导数：")
            print(sess.run(g, feed_dict={goal: goal_numpy}))

        if is_constant:
            sess.run(train_optimizer)
        else:
            sess.run(train_optimizer, feed_dict={goal: goal_numpy})

    print('\n当前运行时间：' + str((datetime.datetime.now() - begin)))



