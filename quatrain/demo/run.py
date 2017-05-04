#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import data_util
from model import neural_network

# load data
batch_size = 1
dataset = '../dataset/'
poetry_file = dataset + 'qtrain'
poetrys = data_util.read_data(poetry_file)
x_batches,y_batches,words,n_chunk = data_util.get_batches(poetrys,batch_size = 64)
word_num_map = dict(zip(words, range(len(words))))  # {"人":5,""}
voca_size = len(words)
print("------------------data process okay---------------------------------------")
#---------------------------------------RNN--------------------------------------#
 
input_data = tf.placeholder(tf.int32, [batch_size, None])    # [64,None]
output_targets = tf.placeholder(tf.int32, [batch_size, None])# [64,None]


def get_word(weights,vocabulary):
	index = weights.argmax(axis = 1)
	return vocabulary[int(index)]

def to_word(weights,vocabulary):
	t = np.cumsum(weights)
	# print("t = ",t.shape)  # (6111,)
	s = np.sum(weights)
	# print("s = ",s)  # s = 1.0
	sample = int(np.searchsorted(t, np.random.rand(1)*s))
	# print("sample = ",sample)  # 随机数 从t中随机抽取
	# print("words = ",len(words))  # 6110
	return vocabulary[sample]   # 从诗歌汉子数据集中选取一个返回

# 使用训练完成的模型
def gen_poetry():
	_, last_state, probs, cell, initial_state = neural_network(input_data = input_data,
															 output_targets = output_targets,
															 batch_size = batch_size,
															 voca_size = voca_size)
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		# init = tf.initialize_all_variables()
		sess.run(init)
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(sess,'../model/poetry.data')
		print("restore model successfully !")
		state_ = sess.run(cell.zero_state(1, tf.float32))
		x = np.array([list(map(word_num_map.get, '['))])  # x.shape = (1,1)
		# print("x = ",x) # x = [[2]]
		[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})

		word = to_word(probs_,vocabulary = words)
		# word = get_word(probs_)
		print("first word = ",word,words[-1])
		#word = words[np.argmax(probs_)]
		poem = ''
		i = 0
		while word != ']':
			if word != '[':
				poem += word
			x = np.zeros((1,1))
			x[0,0] = word_num_map[word]
			[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
			# word = to_word(probs_,vocabulary = words)
			word = get_word(probs_,vocabulary = words)

			i = i + 1
			if i > 200:
				break
		return poem


def gen_poetry_with_head(head):
	def to_word(weights):
		t = np.cumsum(weights)
		s = np.sum(weights)
		sample = int(np.searchsorted(t, np.random.rand(1)*s))
		return words[sample]
 
	_, last_state, probs, cell, initial_state = neural_network()
 
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
 
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(sess,'./model/poetry.data')	
 
		state_ = sess.run(cell.zero_state(1, tf.float32))
		poem = ''
		i = 0
		for word in head:
			while word != '，' and word != '。':
				poem += word
				x = np.array([list(map(word_num_map.get, word))])
				[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
				# word = to_word(probs_)
				word = get_word(probs_)
				time.sleep(1)
			if i % 2 == 0:
				poem += '，'
			else:
				poem += '。'
			i += 1
		return poem
 
# print(gen_poetry_with_head('一二三四'))

if __name__ == '__main__':
	print("-----------------------生成古诗---------------------------#")
	poem = gen_poetry()
	poem = poem.split('.')
	print("-"*50)
	for p in poem:
		print(p)
	print('-'*50)