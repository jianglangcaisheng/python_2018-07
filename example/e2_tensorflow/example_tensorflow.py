
import tensorflow as tf

with tf.Session() as sess:

    variable_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variable_names)
    for k, v in zip(variable_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        print(v)