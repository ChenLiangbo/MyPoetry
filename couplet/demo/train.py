#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
from model import Seq2Seq
import data as data_util

ddir = '../data/'
ckpt = '../ckpt/'

xdata = np.load(ddir + 'xdata.npy')
ydata = np.load(ddir + 'ydata.npy')
print("xdata = ",xdata.shape)  # (246152, 26)
print("ydata = ",ydata.shape)  # (246152, 26)


print(xdata[10])
print(ydata[10])
print("-"*80)
shape = xdata.shape
# train validate,test
dsplit = [9.9,0.05,0.05]
'''
trainX,trainY,validX,validY,testX,testY= data_util.split_data(xdata,ydata,True,dsplit)
print("train = ",trainX.shape,"validate = ",validX.shape,"test = ",testX.shape)
np.save(ddir + 'trainX',trainX)
np.save(ddir + 'trainY',trainY)
np.save(ddir + 'testX',testX)
np.save(ddir + 'testY',testY)
np.save(ddir + 'validX',validX)
np.save(ddir + 'validY',validY)
'''
trainX = np.load(ddir + 'trainX.npy')
trainY = np.load(ddir + 'trainY.npy')
testX = np.load(ddir + 'testX.npy')
testY = np.load(ddir + 'testY.npy')
validX = np.load(ddir + 'validX.npy')
validY = np.load(ddir + 'validY.npy')


vocabulary = data_util.read_vocabulary(ddir)
word_to_num = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
num_to_word = dict(zip(range(len(vocabulary)),vocabulary))

xseq_len = xdata.shape[-1]
yseq_len = ydata.shape[-1]
print("xseq_len = ",xseq_len,"yseq_len = ",yseq_len)
batch_size = 64
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
	num_layers  = 3
	)

val_batch_gen = data_util.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_util.rand_batch_gen(trainX, trainY, batch_size)


sess = model.train(train_batch_gen, val_batch_gen)
