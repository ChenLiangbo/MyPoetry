#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
from model import Seq2Seq
import data as data_util

ddir = './data/'
ckpt = './ckpt/'

xdata = np.load(ddir + 'xdata.npy')
ydata = np.load(ddir + 'ydata.npy')
print("xdata = ",xdata.shape)  # (246152, 26)
print("ydata = ",ydata.shape)  # (246152, 26)
print(xdata[10])
print(ydata[10])
print("-"*80)
shape = xdata.shape
# train validate,test
dsplit = [9.5,0.2,0.3]
trainX,trainY,validX,validY,testX,testY= data_util.split_data(xdata,ydata,False,dsplit)
print("train = ",trainX.shape,"validate = ",validX.shape,"test = ",testX.shape)
vocabulary = data_util.read_vocabulary(ddir)

word_to_num = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
num_to_word = dict(zip(range(len(vocabulary)),vocabulary))

xseq_len = xdata.shape[-1]
yseq_len = ydata.shape[-1]
print("xseq_len = ",xseq_len,"yseq_len = ",yseq_len)
batch_size = 32
# xvocab_size = len(metadata['idx2w'])  
xvocab_size = len(vocabulary)
yvocab_size = xvocab_size
emb_dim = 1024

model = Seq2Seq(xseq_len = xseq_len,
	yseq_len    = yseq_len,
	xvocab_size = xvocab_size,
	yvocab_size = yvocab_size,
	ckpt_path   = ckpt,
	emb_dim     = emb_dim,
	num_layers  = 2
	)

val_batch_gen = data_util.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_util.rand_batch_gen(trainX, trainY, batch_size)


sess = model.train(train_batch_gen, val_batch_gen)
