#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import pickle
temp = '../data/'

s = [0]*10
s1 = [2,3,4,5,3]
s[0:len(s1)] = s1
print("s = ",s)
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