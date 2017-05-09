#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
from seq2seq import SimpleSeq2Seq
# from seq2seq import Seq2Seq
# from seq2seq import AttentionSeq2Seq

input_length = 10
input_dim = 2

output_length = 8
output_dim = 3

samples = 100

x = np.random.random((samples, input_length, input_dim))
y = np.random.random((samples, output_length, output_dim))


model = SimpleSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)
model.compile(loss='mse', optimizer='sgd')
model.fit(x, y, nb_epoch=1)
