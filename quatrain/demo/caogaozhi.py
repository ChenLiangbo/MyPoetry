#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import pickle
temp = '../data/'
'''
s =  '偶寻半开梅,闲倚一竿竹,儿童不知春'
print("s = ",len(s))


fpx = open(temp + 's.pkl','wb')
pickle.dump(s,fpx)
fpx.close()
'''
fpx = open(temp + 's.pkl','rb')
s = pickle.load(fpx)
fpx.close()
print("s = ",s)