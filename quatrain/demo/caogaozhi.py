#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import pickle
temp = '../data/'

batch_size = 64
xdata = np.load('../data/xdata.npy')
ydata = np.load('../data/ydata.npy')
xshape = xdata.shape
yshape = ydata.shape

x = list(range(0,xshape[0]-batch_size ,batch_size))
y = list(range(batch_size,xshape[0],batch_size))

print("x = ",len(x),len(y))
batch = list(zip(x,y))
print(batch[0],batch[-1])


'''
s =  '偶寻半开梅,闲倚一竿竹,儿童不知春'
print("s = ",len(s))


fpx = open(temp + 's.pkl','wb')
pickle.dump(s,fpx)
fpx.close()

fpx = open(temp + 's.pkl','rb')
s = pickle.load(fpx)
fpx.close()
print("s = ",s)
'''
# import data_util

# dataset = '../dataset/'
# temp = '../data/'
# poetry_file = dataset + 'qtrain'
# origin_poetrys = data_util.read_data(poetry_file)
# vocabulary,word_num_map = data_util.get_vocabulary(origin_poetrys)

# w = ''

# n = data_util.string_to_num(w,word_num_map)
# print("n = ",n)