
import tensorflow as tf
# zero_out_module = tf.load_op_library('/home/visg01/tensorflow1/bazel-bin/tensorflow/core/user_ops/zero_out.so')
# zero_out = zero_out_module.zero_out

import time
import datetime


b = tf.Variable (1.57)
g = tf.gradients(tf.sin(b), [b])

addone_module = tf.load_op_library('/home/visg01/tensorflow1/bazel-bin/tensorflow/core/u04071356_sin/cuda_op_kernel.so')
h = tf.gradients(addone_module.user_sin(b), [b])

c= addone_module.user_sin(b)
# c= addone_module.user_sin(b)

loss = tf.reduce_sum(tf.square(c))
train_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, gate_gradients=0)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    print(sess.run(g))
    print(sess.run(h))

    for i in range(100):
        sess.run(train_optimizer)

        print("\n"+"data")
        print(sess.run(b))
        print("gradients")
        print(sess.run(g))
        print(sess.run(h))