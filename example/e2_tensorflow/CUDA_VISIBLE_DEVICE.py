
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

a = tf.Variable(1.)

train_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(a)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range (100000):
        if i % 100 == 0:
            print(i)
        sess.run(train_optimizer)
    print(sess.run(a))