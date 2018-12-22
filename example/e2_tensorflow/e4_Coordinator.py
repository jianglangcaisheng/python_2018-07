
import tensorflow as tf
import numpy as np

import time
import datetime

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TYPE = tf.float32

strategy = 1

a = tf.Variable(0.)
b = tf.Variable(0.)

goal1 = tf.constant(10.)
goal2 = tf.constant(10.)


loss1 = tf.square(a - goal1)
loss2 = tf.square(b - goal2)

train_optimizer_1 = tf.train.GradientDescentOptimizer(0.1).minimize(loss1, gate_gradients=0)
train_optimizer_2 = tf.train.GradientDescentOptimizer(0.1).minimize(loss2, gate_gradients=0)


init = tf.initialize_all_variables()
with tf.Session() as sess:
    train_times = 15000

    begin = datetime.datetime.now()

    if strategy == 1:
        sess.run(init)
        print(sess.run(a))
        print(sess.run(b))
        print('\n当前运行时间：' + str((datetime.datetime.now() - begin)))
        if 1:
            def MyLoop(coord):
                train_times_already = 0
                while not coord.should_stop():
                    sess.run(train_optimizer_1)
                    train_times_already = train_times_already + 1
                    if train_times_already >= train_times:
                        coord.request_stop()


            def MyLoop2(coord):
                train_times_already = 0
                while not coord.should_stop():
                    sess.run(train_optimizer_2)
                    train_times_already = train_times_already + 1
                    if train_times_already >= train_times:
                        coord.request_stop()

            coord = tf.train.Coordinator()
            coord2 = tf.train.Coordinator()

            # Create 10 threads that run 'MyLoop()'
            import threading

            threads = [threading.Thread(target=MyLoop, args=(coord, )) for i in range(1)]
            threads.append(threading.Thread(target=MyLoop2, args=(coord2, )))

        print('\n当前运行时间：' + str((datetime.datetime.now() - begin)))
        # Start the threads and wait for all of them to stop.
        for t in threads:
            t.start()

    #
    # for i in range(train_times):
    #
    #     sess.run(train_optimizer_1)
    #     sess.run(train_optimizer_2)
    if strategy == 1:
        coord.join(threads)

    print('\n当前运行时间：' + str((datetime.datetime.now() - begin)))
    print(sess.run(a))
    print(sess.run(b))



