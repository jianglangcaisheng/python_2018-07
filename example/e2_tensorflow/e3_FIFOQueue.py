
import tensorflow as tf
import numpy as np

import time
import datetime

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TYPE = tf.float32

strategy = 1

a = tf.Variable(0.)
shape = (2400, 100)
if strategy == 0:
    goal = tf.ones(shape=shape)
elif strategy == 1:
    goal = tf.placeholder(shape=shape, dtype=TYPE)
    goal_numpy = np.ones(shape=shape)
elif strategy == 2:
    goal_queue = tf.FIFOQueue(1500, TYPE, shapes=shape)
    goal_placeholder = tf.placeholder(dtype=TYPE, shape=shape)
    init_queue = goal_queue.enqueue(goal_placeholder)
    goal = goal_queue.dequeue()

    goal_numpy = np.ones(shape=shape)
else:
    assert False, "NULL"


loss = tf.square(a - tf.reduce_sum(goal))
# g = tf.gradients(loss, a)

train_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss, gate_gradients=0)


init = tf.initialize_all_variables()
with tf.Session() as sess:


    begin = datetime.datetime.now()

    if strategy == 0:
        sess.run(init)
    elif strategy == 1:
        sess.run(init, feed_dict={goal: goal_numpy})
    elif strategy == 2:
        sess.run(init)
        print('\n当前运行时间：' + str((datetime.datetime.now() - begin)))
        if 0:
            for i_data in range(1500):
                sess.run(init_queue, feed_dict={goal_placeholder: goal_numpy})
        else:
            if 1:
                def MyLoop(coord):
                    queue_length = 0
                    while not coord.should_stop():
                        sess.run(init_queue, feed_dict={goal_placeholder: goal_numpy})
                        queue_length = queue_length + 1
                        if queue_length == 1500:
                            coord.request_stop()

                coord = tf.train.Coordinator()

                # Create 10 threads that run 'MyLoop()'
                import threading

                threads = [threading.Thread(target=MyLoop, args=(coord, )) for i in range(1)]

            # Start the threads and wait for all of them to stop.
            for t in threads:
                t.start()
    else:
        assert False, "null"

    print('\n当前运行时间：' + str((datetime.datetime.now() - begin)))

    train_times = 1500
    for i in range(train_times):

        if strategy == 0:
            sess.run(train_optimizer)
        elif strategy == 1:
            sess.run(train_optimizer, feed_dict={goal: goal_numpy})
        elif strategy == 2:
            sess.run(train_optimizer)
    if strategy == 2:
        coord.join(threads)

    print('\n当前运行时间：' + str((datetime.datetime.now() - begin)))



