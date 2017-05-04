#!usr/bin/env/python 
# -*- coding: utf-8 -*-

class Config(object):
	def __init__(self,):
		super(Config,self).__init__()
		self.batch_size     = 64
		self.rnn_size       = 256
		self.dimensionality = 128

configer = Config()