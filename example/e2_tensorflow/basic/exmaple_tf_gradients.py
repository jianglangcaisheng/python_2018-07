
import tensorflow as tf
# zero_out_module = tf.load_op_library('/home/visg01/tensorflow1/bazel-bin/tensorflow/core/user_ops/zero_out.so')
# zero_out = zero_out_module.zero_out

import time
import datetime


# b = tf.Variable (1.57)
# g = tf.gradients(tf.sin(b), b)

a = tf.ones(shape=[100])
b = tf.nn.softmax(a)
g = tf.gradients(b, a)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(g))
