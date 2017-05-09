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
shape = xdata.shape
# train validate,test
dsplit = [0.95,0.02,0.03]
length = int(dsplit[0]*shape[0]/10)
length1= int(dsplit[1]*shape[0]/10)

trainX = xdata[:length,:]
trainY = ydata[:length,:]
validX = xdata[length:length+length1,:]
validY = ydata[length:length+length1,:]
testX  = xdata[length+length1:,:]
testY  = ydata[length+length1:,:]
print("train = ",trainX.shape,"validate = ",validX.shape,"test = ",testX.shape)


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

sess = model.restore_last_session()

predictY = model.predict(sess,testX)

for i in range(100):
	real = data_util.num_to_poem(testY[i])
	fake = data_util.num_to_poem(predictY[i])
	print("real poem is : ",real)
	print("fake poem is : ",fake)
	print("-"*80)