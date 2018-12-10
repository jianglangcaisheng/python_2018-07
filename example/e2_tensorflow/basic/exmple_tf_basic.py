
import tensorflow as tf

import time
import datetime

a = tf.Variable([[1., 2., 3., 4., 5.],
                 [6., 7., 8., 9., 10.]])
a2 = tf.Variable([[1., 2., 3., 4., 5.],
                  [6., 7., 8., 9., 10.]])
a3 = tf.Variable([[1., 0., 1., 0., 1.],
                  [1., 0., 1., 0., 1.]])
b1 = tf.Variable([[1.]])
b2 = tf.constant([[1.]])

variable_1 = tf.Variable(1.)
variable_2_5 = tf.Variable([[1., 2., 3., 4., 5.],
                        [6., 7., 8., 9., 10.]])

# sqrt
if 0:
    b = tf.pow(a, 2)
    # [[   1.    4.    9.   16.   25.]
    #  [  36.   49.   64.   81.  100.]]
    b = tf.pow(-a, 2)
    # [[   1.    4.    9.   16.   25.]
    #  [  36.   49.   64.   81.  100.]]

# sqrt
if 0:
    b = tf.sqrt(tf.multiply(a[0:1, :], a[0:1, :]) +
                tf.multiply(a[1:2, :], a[1:2, :]))

# 筛选
if 0:
    b3 = a * tf.cast(a > 5., tf.float32)

# 多变量
if 0:
    a1 = tf.Variable(1.)
    a2 = tf.Variable(1.)
    b = a1 + a2
    b2 = tf.sqrt(a1 * a1 + a2 * a2)

# 整除
if 0:
    a1 = tf.Variable(1.5)
    b = tf.mod(a1, 1.2)

# concat
if 0:
    c1 = tf.concat([b1, b2], axis=1)
    # [[  4.89378602e-38   2.00000000e+00]]

# atan2
if 0:
    c1 = tf.atan2(b1, b2)
    # [-\pi, \pi]

# 不可slice赋值
if 0:
    a[0, 0] = a2[0, 0]        # Variable 不支持

# 取余
if 1:
    # b = tf.mod(variable_1, 0.3)     # No gradient defined for operation 'FloorMod' (op type: FloorMod)
    b = variable_1 - tf.floor(variable_1 / 0.3) * 0.3

loss = tf.reduce_sum(tf.square(b))
train_optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss, gate_gradients=0)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    print(sess.run(variable_1))
    print(sess.run(b))

    for i in range(20000):
        sess.run(train_optimizer)

    print(sess.run(variable_1))
    print(sess.run(b))

