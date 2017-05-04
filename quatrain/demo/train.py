#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import data_util
from model import neural_network

batch_size = 64
dataset = '../dataset/'
poetry_file = dataset + 'qtrain'
poetrys = data_util.read_data(poetry_file)
x_batches,y_batches,words,n_chunk = data_util.get_batches(poetrys,batch_size = 64)
voca_size = len(words)
print("------------------data process okay---------------------------------------")

input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])


def train_neural_network(epoch):
	logits, last_state, _, _, _ = neural_network(input_data = input_data,
												output_targets = output_targets,
												batch_size = batch_size,
												voca_size = voca_size)
	# logits shape = (?,voca_size)
	targets = tf.reshape(output_targets, [-1])
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
		try:
			saver.restore(sess,'../model/poetry.data')
			print("load model okay,train again!")
		except Exception as ex:
			print("[Exception Information] ",str(ex))

 
		for epoch in range(epoch):
			sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
			n = 0
			for batche in range(n_chunk):
				train_loss, _ , _ = sess.run([cost, last_state, train_op],
				              feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
				n += 1
				if batche % 100 == 0:
					print("epoch =%d, batche = %d, train_loss = %f" % (epoch, batche, train_loss))
				# break
			saver.save(sess, '../model/poetry.data')
 
if __name__ == '__main__':
	train_neural_network(epoch = 1)