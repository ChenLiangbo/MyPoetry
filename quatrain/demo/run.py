#!usr/bin/env/python 
# -*- coding: utf-8 -*-

# http://blog.topspeedsnail.com/archives/10542
# sudo python3 -m pip install tensorflow==0.12.0
# http://blog.csdn.net/u014365862/article/details/53868544
import collections
import numpy as np
import tensorflow as tf
import data_util
#-------------------------------数据预处理---------------------------#
batch_size = 1
dataset = '../dataset/'
poetry_file = dataset + 'qtrain'
poetrys = data_util.read_data(poetry_file)
x_batches,y_batches,words,n_chunk = data_util.get_batches(poetrys,batch_size = 64)
word_num_map = dict(zip(words, range(len(words))))  # {"人":5,""}
print("------------------data process okay---------------------------------------")
#---------------------------------------RNN--------------------------------------#
 
input_data = tf.placeholder(tf.int32, [batch_size, None])    # [64,None]
output_targets = tf.placeholder(tf.int32, [batch_size, None])# [64,None]
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
	if model == 'rnn':
		cell_fun = tf.nn.rnn_cell.BasicRNNCell
	elif model == 'gru':
		cell_fun = tf.nn.rnn_cell.GRUCell
	elif model == 'lstm':
		cell_fun = tf.nn.rnn_cell.BasicLSTMCell
 
	cell = cell_fun(rnn_size, state_is_tuple=True)
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
 
	initial_state = cell.zero_state(batch_size, tf.float32)
 
	with tf.variable_scope('rnnlm'):
		softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1])
		softmax_b = tf.get_variable("softmax_b", [len(words)+1])
		with tf.device("/cpu:0"):
			# embedding = (6111,128) inputs = (1,?,128)
			embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data)
			# print("inputs.get_shape = ",inputs.get_shape())
 
	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
	# print("outputs.get_shape = ",outputs.get_shape()) # (1,?,128)
	output = tf.reshape(outputs,[-1, rnn_size])       # (?,128)
	# print("output.get_shape = ",output.get_shape())

 
	logits = tf.matmul(output, softmax_w) + softmax_b  # (?,6111)
	# print("logits.get_shape = ",logits.get_shape())
	probs = tf.nn.softmax(logits)
	# print("probs.get_shape = ",probs.get_shape())       #(?,6111)
	return logits, last_state, probs, cell, initial_state

print("#-------------------------------生成古诗---------------------------------#")
# array([[1, 2, 3, 4],
#        [3, 4, 5, 6]])                                  
# array([ 1,  3,  6, 10, 13, 17, 22, 28])  np.cumsum(a)
# (2,4) -> (8,)
def get_word(weights):
	index = weights.argmax(axis = 1)
	return words[int(index)]

# 使用训练完成的模型
def gen_poetry():
	def to_word(weights):
		t = np.cumsum(weights)
		# print("t = ",t.shape)  # (6111,)
		s = np.sum(weights)
		# print("s = ",s)  # s = 1.0
		sample = int(np.searchsorted(t, np.random.rand(1)*s))
		# print("sample = ",sample)  # 随机数 从t中随机抽取
		# print("words = ",len(words))  # 6110
		return words[sample]   # 从诗歌汉子数据集中选取一个返回
 
	_, last_state, probs, cell, initial_state = neural_network()
 
	with tf.Session() as sess:

		init = tf.global_variables_initializer()
		# init = tf.initialize_all_variables()
		sess.run(init)
 
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(sess,'./model/poetry.data')
		print("restore model successfully !")
		state_ = sess.run(cell.zero_state(1, tf.float32))
		x = np.array([list(map(word_num_map.get, '['))])  # x.shape = (1,1)
		# print("x = ",x) # x = [[2]]
		[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})

		word = to_word(probs_)
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
			# word = to_word(probs_)
			word = get_word(probs_)
			# print("word = ",word," poem = ",poem)
			#word = words[np.argmax(probs_)]
			# break
			i = i + 1
			if i > 200:
				break
		return poem
poem = gen_poetry()
poem = poem.split('。')
print("-"*50)
for p in poem:
	print(p)
print('-'*50)


print("--------------------------------生成藏头诗--------------------------------#")
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