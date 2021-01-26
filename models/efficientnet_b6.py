from keras_efficientnets import EfficientNetB7
from keras_efficientnets import EfficientNetB5
from keras_efficientnets import EfficientNetB6
import efficientnet.keras as efn
from tensorflow.contrib.layers import batch_norm, flatten
from tflearn.layers.conv import global_avg_pool




import tensorflow as tf
import numpy as np


class EfficientNetb6p:
	def __init__(self, features, phase,opt=None):
		self.opt = opt


		# Zero-mean input
		with tf.name_scope('preprocess') as scope:
			print("before normalization sssssssssssssssssssssssgugugugugugugugu")
			print(features.shape)
			rgb_mean = [102.153, 100.801, 105.317]
			rgb_std = [61.305, 59.688, 56.209]
			bb_mean = [4]
			bb_std = [18.679]
			mean = rgb_mean*self.opt.n_rgbs_per_sample + bb_mean*self.opt.n_bbs_per_sample
			mean = tf.constant(mean, dtype=tf.float32, shape=[1, 1, 1, self.opt.input_channel_dim], name='img_mean')
			std = rgb_std*self.opt.n_rgbs_per_sample + bb_std*self.opt.n_bbs_per_sample
			std = tf.constant(std, dtype=tf.float32, shape=[1, 1, 1, self.opt.input_channel_dim], name='img_std')
			features = (features-mean)/std
			print("after nornmalizationsssssssssssssssssssssssgugugugugugugugu")
			print(features.shape)


		inp = features
		#inp = tf.reshape(features, shape=[135, 355, 3])
		# base = EFNS[ef](input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
		# x = base(inp)
		# x = tf.keras.layers.GlobalAveragePooling2D()(x)
		# x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
		# model = tf.keras.Model(inputs=inp, outputs=x)






		#######################################################################

		# Defining the model
		base = EfficientNetB6(include_top=False, weights=None,input_shape=(130, 355, 12),
									classes=2)

		# Adding the final layers to the above base models where the actual classification is done in the dense layers
		x = base(inp)
		axes = [2, 3]
		x = tf.reduce_mean(x, axes, keepdims=True)
		x = flatten(x)
		
		x = tf.layers.dense(inputs=x, use_bias=False, units=256)
		x = tf.layers.dense(inputs=x, use_bias=False, units=128)
		x = tf.layers.dense(inputs=x, use_bias=False, units=2)
		#x = tf.keras.layers.GlobalAveragePooling2D()(x)
		#x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
		#model = tf.keras.Model(inputs=inp, outputs=x)
		self.logits = tf.identity(x, 'final_dense')


		# model = Sequential()
		# model.add(x)
		# model.add(Flatten())
		# model.add(Dense(1024, activation=('relu'), input_dim=512))
		#
		# model.add(Dense(512, activation=('relu')))
		# model.add(Dense(256, activation=('relu')))
		# # model.add(Dropout(.3))
		# model.add(Dense(128, activation=('relu')))
		# # model.add(Dropout(.2))
		# l = model.add(Dense(2, activation=('softmax')))

		#self.logits = tf.identity(l, 'final_dense')




	def fc_layers(self, keep_prob):
		self.fc_parameters = []

		# fc1
		with tf.variable_scope('fc1') as scope:
			shape = int(np.prod(self.pool5.get_shape()[1:]))
			fc1w = tf.get_variable('fc_weights', shape=[shape, 100], dtype=tf.float32,
									initializer=self.initializer())
			fc1b = tf.get_variable('fc_biases', [100], tf.float32,
									tf.constant_initializer(self.opt.init_bias_value))
			pool5_flat = tf.reshape(self.pool5, [-1, shape])
			self.fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
			self.fc1 = tf.nn.relu(self.fc1l)
			self.fc_parameters += [fc1w, fc1b]

			if self.opt.keep_prob < 1.0:
				self.fc1 = tf.nn.dropout(self.fc1, keep_prob)
			else:
				print("Dropout is not used")

		# fc2
		with tf.variable_scope('fc2') as scope:
			fc2w = tf.get_variable('fc_weights', shape=[100, 2], dtype=tf.float32,
									initializer=self.initializer())
			fc2b = tf.get_variable('fc_biases', [2], tf.float32,
									tf.constant_initializer(self.opt.init_bias_value))
			self.logits = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
			self.fc_parameters += [fc2w, fc2b]

			self.probs = tf.nn.softmax(self.logits)

def xavier_init(self):
		xavier_init =  tf.contrib.layers.xavier_initializer()
		return xavier_init



