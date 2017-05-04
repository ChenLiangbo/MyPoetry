#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import tensorflow as tf

# 定义RNN
def neural_network(input_data,output_targets,batch_size,voca_size,model='lstm', rnn_size=128, num_layers=2):
	print("-"*40)
	if model == 'rnn':
		cell_fun = tf.nn.rnn_cell.BasicRNNCell
	elif model == 'gru':
		cell_fun = tf.nn.rnn_cell.GRUCell
	elif model == 'lstm':
		cell_fun = tf.nn.rnn_cell.BasicLSTMCell
 
	cell = cell_fun(rnn_size, state_is_tuple=True)
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
	# cell.output_size = 128
	# cell.state_size :(LSTMStateTuple(c=128, h=128), LSTMStateTuple(c=128, h=128))

	initial_state = cell.zero_state(batch_size, tf.float32)
	with tf.variable_scope('rnnlm'):
		softmax_w = tf.get_variable("softmax_w", [rnn_size, voca_size+1])
		softmax_b = tf.get_variable("softmax_b", [voca_size+1])
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [voca_size+1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data)
 
	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
	output = tf.reshape(outputs,[-1, rnn_size])
	print("output = ",output.get_shape())
	# print("last_state = ",dir(last_state))
 
	logits = tf.matmul(output, softmax_w) + softmax_b
	print("logits = ",logits.get_shape())
	probs = tf.nn.softmax(logits)
	print("probs = ",probs.get_shape())
	print("-"*40)
	return logits, last_state, probs, cell, initial_state