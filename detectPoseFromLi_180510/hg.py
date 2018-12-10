import tensorflow as tf
import numpy as np

class initPara():
	def __init__(self, idx, shape=None):

		import os
		pathName_file = os.path.dirname(__file__) + '/para/' + str(idx) + '.npy'

		data = np.load(pathName_file)
		# data = np.load('para/' + str(idx) + '.npy')
		self.data = tf.Variable(data, dtype=tf.float32)
	def __call__(self, shape, dtype, partition_info):
		return self.data

class Model():

	def __init__(self):
		self._createModel()

	def _getPara(self, idx, shape=None):
		import os
		pathName_file = os.path.dirname(__file__) + '/para/' + str(idx) + '.npy'
		return np.load(pathName_file)
		# return np.load('para/' + str(idx) + '.npy')

	def _conv(self, inputs, weights, bias, stride, pad, name=None):
		if pad != 0:
			pad = tf.pad(inputs, [[0,0], [pad,pad], [pad,pad], [0,0]])
		else:
			pad = inputs
		w = tf.Variable(weights, dtype=tf.float32)
		b = tf.Variable(bias, dtype=tf.float32)
		conv = tf.nn.conv2d(pad, w, [1, stride, stride, 1], padding='VALID', data_format='NHWC')
		if name != None:
			return tf.nn.bias_add(conv, b, name=name)
		else:
			return tf.nn.bias_add(conv, b)

	def _residual(self, inputs, numOut, idx):
		norm1 = tf.layers.batch_normalization(inputs, training=True, epsilon=1e-5, gamma_initializer=initPara(idx+0), beta_initializer=initPara(idx+1), momentum=0.99)
		relu1 = tf.nn.relu(norm1)
		conv1 = self._conv(relu1, np.transpose(self._getPara(idx+2), (2, 3, 1, 0)), self._getPara(idx+3), 1, 0)

		norm2 = tf.layers.batch_normalization(conv1, training=True, epsilon=1e-5, gamma_initializer=initPara(idx+4), beta_initializer=initPara(idx+5), momentum=0.99)
		relu2 = tf.nn.relu(norm2)
		conv2 = self._conv(relu2, np.transpose(self._getPara(idx+6), (2, 3, 1, 0)), self._getPara(idx+7), 1, 1)

		norm3 = tf.layers.batch_normalization(conv2, training=True, epsilon=1e-5, gamma_initializer=initPara(idx+8), beta_initializer=initPara(idx+9), momentum=0.99)
		relu3 = tf.nn.relu(norm3)
		conv3 = self._conv(relu3, np.transpose(self._getPara(idx+10), (2, 3, 1, 0)), self._getPara(idx+11), 1, 0)

		if inputs.get_shape().as_list()[3] != numOut:
			conv4 = self._conv(inputs, np.transpose(self._getPara(idx+12), (2, 3, 1, 0)), self._getPara(idx+13), 1, 0)
			return tf.add_n([conv3, conv4])
		else:
			return tf.add_n([conv3, inputs])

	def _hourglass(self, inputs, numOut, nPool, idx):
		r1   = self._residual(inputs, numOut, idx)

		pool = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
		r2   = self._residual(pool, numOut, idx+12)
		if nPool > 1:
			r3 = self._hourglass(r2, numOut, nPool-1, idx+24)
		else:
			r3 = self._residual(r2, numOut, idx+24)
		r4 = self._residual(r3, numOut, idx + 36 * nPool)
		r5 = tf.image.resize_nearest_neighbor(r4, tf.shape(r4)[1:3]*2)

		return tf.add_n([r1, r5])

	def _createModel(self):
		self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='inputs')
		self.labels = tf.placeholder(dtype=tf.float32, shape=[4, None, 64, 64, 16], name='labels')

		conv = self._conv(self.inputs, np.transpose(self._getPara(0), (2, 3, 1, 0)), self._getPara(1), 2, 3)
		norm = tf.layers.batch_normalization(conv, training=True, epsilon=1e-5, gamma_initializer=initPara(2), beta_initializer=initPara(3), momentum=0.99)
		relu = tf.nn.relu(norm)
		r1   = self._residual(relu, 128, 4)
		pool = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding='VALID')
		r4   = self._residual(pool, 128, 18)
		r5   = self._residual(r4, 256, 30)

		inter = r5
		out = []
		idx = 44
		for i in range(4):
			h = self._hourglass(inter, 256, 4, idx)
			r = self._residual(h, 256, idx+156)
			c = self._conv(r, np.transpose(self._getPara(idx+168), (2, 3, 1, 0)), self._getPara(idx+169), 1, 0)
			n = tf.layers.batch_normalization(c, training=True, epsilon=1e-5, gamma_initializer=initPara(idx+170), beta_initializer=initPara(idx+171), momentum=0.99)
			relu = tf.nn.relu(n)
			tmpOut = self._conv(relu, np.transpose(self._getPara(idx+172), (2, 3, 1, 0)), self._getPara(idx+173), 1, 0, name='output%d'%(i+1))
			out.append(tmpOut)
			if i < 3:
				ll_ = self._conv(relu, np.transpose(self._getPara(idx+174), (2, 3, 1, 0)), self._getPara(idx+175), 1, 0)
				tmpOut_ = self._conv(tmpOut, np.transpose(self._getPara(idx+176), (2, 3, 1, 0)), self._getPara(idx+177), 1, 0)
				inter = inter + ll_ + tmpOut_
			idx += 178

		self.output = out
		self.loss = tf.reduce_mean(tf.square(self.output[0] - self.labels[0])) + tf.reduce_mean(tf.square(self.output[1] - self.labels[1])) + tf.reduce_mean(tf.square(self.output[2] - self.labels[2])) + tf.reduce_mean(tf.square(self.output[3] - self.labels[3]))

		#rmsprop = tf.train.RMSPropOptimizer(2.5e-4, 0.99, 0.0, 1e-8)
		#rmsprop = tf.train.GradientDescentOptimizer(2.5e-4)
		rmsprop = tf.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.99, momentum=0.0, epsilon=1e-8, use_locking=False, centered=False)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_op = rmsprop.minimize(self.loss)


if __name__ == '__main__':
	model = Model()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(10):
			inputs = np.transpose(np.load('data/inputs' + str(i) + '.npy'), (0, 2, 3, 1))
			labels = np.transpose(np.load('data/labels' + str(i) + '.npy'), (0, 1, 3, 4, 2))
			_, loss = sess.run([model.train_op, model.loss], feed_dict={model.inputs:inputs, model.labels:labels})
			print(i, ':', float(loss))