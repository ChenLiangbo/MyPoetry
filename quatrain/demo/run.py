#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import MyModel
import data_util
import pickle

data = '../data/'
ckpt = '../ckpt/'

xdata = np.load(data + 'xdata.npy')
ydata = np.load(data + 'ydata.npy')
print("xdata = ",xdata.shape)  # (246152, 26)
print("ydata = ",ydata.shape)  # (246152, 26)
print(xdata[10])
print(ydata[10])
print("-"*80)
shape = xdata.shape
# train validate,test
dsplit = [9.5,0.2,0.3]
trainX,trainY,validX,validY,testX,testY = data_util.split_data(xdata,ydata,dsplit)
print("train = ",trainX.shape,"validate = ",validX.shape,"test = ",testX.shape)

dataset = '../newdata/'
poetry_file = dataset + 'quatrain7'
fpx = open(data + 'keyword_poetrys.pkl','rb')
keyword_poetrys = pickle.load(fpx)
fpx.close()
vocabulary = data_util.get_vocabulary(keyword_poetrys)

word_num_map = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
num_word_map = dict(zip(range(len(vocabulary)),vocabulary))


xseq_len = xdata.shape[-1]
yseq_len = ydata.shape[-1]
batch_size = 32
# xvocab_size = len(metadata['idx2w'])  
xvocab_size = len(vocabulary)
yvocab_size = xvocab_size
emb_dim = 1024


model = MyModel.Seq2Seq(xseq_len    = xseq_len,
                        yseq_len    = yseq_len,
                        xvocab_size = xvocab_size,
                        yvocab_size = yvocab_size,
                        ckpt_path   = ckpt,
                        emb_dim     = emb_dim,
                        num_layers  = 3
                         )
sess = model.restore_last_session()
'''
predictY = model.predict(sess,testX)
print("predictY = ",predictY.shape,"testX = ",testX.shape,"testY = ",testY.shape)
for i in range(10):
    y1 = predictY[i,:]
    y2 = testY[i,:]
    real = data_util.num_to_poem(y2,vocabulary)
    fake = data_util.num_to_poem(y1,vocabulary)
    print("real = ",real)
    print("fake = ",fake)
    print('-'*80)
print('+'*80)
'''

shape = testX.shape
for i in range(100):
      x = testX[i,:].reshape((1,shape[1]))
      y = testY[i,:].reshape((1,ydata.shape[1]))
      #print("x = ",x.shape,"y = ",y.shape)
      py= model.predict_one(sess,x)
      #print("py = ",py)
      #print("y = ",y)
      real = data_util.num_to_poem(y[0],vocabulary)
      fake = data_util.num_to_poem(py[0],vocabulary)
      print("real poem is : ",real)
      print("fake poem is : ",fake)
      print("-"*80)
      # break


