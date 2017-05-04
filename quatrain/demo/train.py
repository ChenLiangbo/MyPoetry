#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import data_util
from model import neural_network
from config import configer

dataset = '../dataset/'
temp = '../data/'
xdata = np.load('../data/xdata.npy')
ydata = np.load('../data/ydata.npy')
xshape = xdata.shape
yshape = ydata.shape
print("xshape = ",xshape,"yshape = ",yshape)

poetry_file = dataset + 'qtrain'
origin_poetrys = data_util.read_data(poetry_file)
vocabulary,word_num_map = data_util.get_vocabulary(origin_poetrys)
voca_size = len(vocabulary)
print("voca_size = ",voca_size)
print("------------------data process okay---------------------------------------")

batch_size = configer.batch_size
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])


def train_neural_network():
	logits, last_state, _, _, _ = neural_network(input_data,output_targets,voca_size)
	print("logits = ",logits.get_shape())
	# logits shape = (?,voca_size)
	targets = tf.reshape(output_targets, [-1])  # 错误在这里 如何修改？
	print("targets = ",targets.get_shape())
	loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], voca_size)
	cost = tf.reduce_mean(loss)
	learning_rate = tf.Variable(0.0, trainable=False)
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.apply_gradients(zip(grads, tvars))
 
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		# init = tf.initialize_all_variables()
		sess.run(init)
		saver = tf.train.Saver()
		# try:
		# 	saver.restore(sess,'../model/poetry.data')
		# 	print("load model okay,train again!")
		# except Exception as ex:
		# 	print("[Exception Information] ",str(ex))
		
		x = list(range(0,xshape[0]-batch_size ,batch_size))
		y = list(range(batch_size,xshape[0],batch_size))
		batches = list(zip(x,y))
 
		for epoch in range(configer.epoch):
			sess.run(tf.assign(learning_rate, 0.022 * (0.97 ** epoch)))
			n = 0
			for start,end in batches:
				x_batch = xdata[start:end,:]
				y_batch = ydata[start:end,:]
				print("x_batch = ",x_batch.shape,"y_batch = ",y_batch.shape)
				train_loss, _ , _ = sess.run([cost, last_state, train_op],
				              feed_dict={input_data: x_batch, output_targets: y_batch})
				n += 1
				if n % 100 == 0:
					print("epoch =%d, step = %d, loss = %f" % (epoch, n, train_loss))
				# break
			saver.save(sess, '../model/poetry.data')
 
if __name__ == '__main__':
	train_neural_network()