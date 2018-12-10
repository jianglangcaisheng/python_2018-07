
import tensorflow as tf
# zero_out_module = tf.load_op_library('/home/visg01/tensorflow1/bazel-bin/tensorflow/core/user_ops/zero_out.so')
# zero_out = zero_out_module.zero_out

import time
import datetime

a1 = tf.Variable([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])

c= tf.user_ops.zero_out (a1, preserve_index=4)

loss=tf.reduce_sum (tf.square(c))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    begin = datetime.datetime.now()
    for i in range(10000):
        sess.run(train_step)
    # print(sess.run(loss))
    # print('当前运行时间：' + str((datetime.datetime.now() - begin)))
    #
    print(sess.run(a1))
    # print(sess.run(c))


