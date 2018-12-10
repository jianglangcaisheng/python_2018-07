
import tensorflow as tf
# zero_out_module = tf.load_op_library('/home/visg01/tensorflow1/bazel-bin/tensorflow/core/user_ops/zero_out.so')
# zero_out = zero_out_module.zero_out

import time
import datetime

a1 = tf.Variable([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])
addone_module = tf.load_op_library('/home/visg01/tensorflow1/bazel-bin/tensorflow/core/u04071356_sin/cuda_op_kernel.so')

c= addone_module.user_sin(a1)

loss = tf.reduce_sum(tf.square(c))
train_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, gate_gradients=0)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)



    begin = datetime.datetime.now()
    for i in range(20000):
        sess.run(train_optimizer)
        # sess.run(c)
    print("\n")
    print('当前运行时间：' + str((datetime.datetime.now() - begin)))
    print(sess.run(c))

