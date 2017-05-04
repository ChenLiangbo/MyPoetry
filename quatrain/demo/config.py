#!usr/bin/env/python 
# -*- coding: utf-8 -*-

class Config(object):
	def __init__(self,):
		super(Config,self).__init__()
		self.batch_size     = 64
		self.rnn_size       = 128
		self.dimensionality = 128
		self.num_layers     = 2
		self.model          = 'lstm'
		self.epoch          = 1

configer = Config()