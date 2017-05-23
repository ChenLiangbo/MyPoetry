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
trainX,trainY,validX,validY,testX,testY= data_util.split_data(xdata,ydata,False,dsplit)
print("train = ",trainX.shape,"validate = ",validX.shape,"test = ",testX.shape)
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

# sess = model.restore_last_session()
sess = model.load_model()

# predictY = model.predict(sess,testX)
# print("predictY = ",predictY.shape,"testX = ",testX.shape,"testY = ",testY.shape)
# for i in range(10):
#     y1 = predictY[i,:]
#     y2 = testY[i,:]
#     real = data_util.decode_to_string(y2,num_to_word)
#     fake = data_util.decode_to_string(y1,num_to_word)
#     print("real = ",real)
#     print("fake = ",fake)
#     print('-'*80)
# print('+'*80)

print("="*40 +'training' + '='*40)
for i in range(20):
      x = trainX[i,:].reshape((1,xdata.shape[1]))
      y = trainY[i,:].reshape((1,ydata.shape[1]))
      #print("x = ",x.shape,"y = ",y.shape)
      py= model.predict_one(sess,x)
      #print("py = ",py)
      #print("y = ",y)
      couplet_up = data_util.decode_to_string(x[0],num_to_word)
      real = data_util.decode_to_string(y[0],num_to_word)
      fake = data_util.decode_to_string(py[0],num_to_word)
      print("couplet_up is   : ",couplet_up)
      print("real couplet is : ",real)
      print("fake couplet is : ",fake)
      print("-"*80)
      # break

print("="*40 +'test' + '='*40)
for i in range(20):
      x = testX[i,:].reshape((1,xdata.shape[1]))
      y = testY[i,:].reshape((1,ydata.shape[1]))
      #print("x = ",x.shape,"y = ",y.shape)
      py= model.predict_one(sess,x)
      #print("py = ",py)
      #print("y = ",y)
      couplet_up = data_util.decode_to_string(x[0],num_to_word)
      real = data_util.decode_to_string(y[0],num_to_word)
      fake = data_util.decode_to_string(py[0],num_to_word)
      print("couplet_up is   : ",couplet_up)
      print("real couplet is : ",real)
      print("fake couplet is : ",fake)
      print("-"*80)
      # break

print("="*40 +'validation' + '='*40)
for i in range(20):
      x = validX[i,:].reshape((1,validX.shape[1]))
      y = validY[i,:].reshape((1,ydata.shape[1]))
      #print("x = ",x.shape,"y = ",y.shape)
      py= model.predict_one(sess,x)
      #print("py = ",py)
      #print("y = ",y)
      couplet_up = data_util.decode_to_string(x[0],num_to_word)
      real = data_util.decode_to_string(y[0],num_to_word)
      fake = data_util.decode_to_string(py[0],num_to_word)
      print("couplet_up is   : ",couplet_up)
      print("real couplet is : ",real)
      print("fake couplet is : ",fake)
      print("-"*80)
      # break