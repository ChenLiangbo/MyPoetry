#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import MyModel
import data_util


data = '../data/'
temp = '../ckpt/'

xdata = np.load(data + 'xdata.npy')
ydata = np.load(data + 'ydata.npy')
print("xdata = ",xdata.shape)  # (246152, 26)
print("ydata = ",ydata.shape)  # (246152, 26)
print(xdata[10])
print(ydata[10])
print("-"*80)


dataset = '../newdata/'
poetry_file = dataset + 'quatrain7'
origin_poetrys = data_util.read_data(poetry_file)
vocabulary = data_util.get_vocabulary(origin_poetrys)
word_num_map = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
num_word_map = dict(zip(range(len(vocabulary)),vocabulary))

xseq_len = xdata.shape[-1]
yseq_len = ydata.shape[-1]
batch_size = 32
# xvocab_size = len(metadata['idx2w'])  
xvocab_size = len(vocabulary)
yvocab_size = xvocab_size
emb_dim = 1024

model = MyModel.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path=ckpt,
                               emb_dim=emb_dim,
                               num_layers=3
                               )
val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
