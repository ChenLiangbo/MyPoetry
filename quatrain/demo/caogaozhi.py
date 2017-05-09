#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import data_util
import numpy as np


if __name__ == '__main__':
	import pickle
	dataset = '../newdata/'
	temp = '../data/'
	poetry_file = dataset + 'quatrain7'

	# keyword_poetrys = keyword_poem(poetry_file) # get keyword from whole poem
	keyword_poetrys = data_util.keyword_line(poetry_file) # get keyword line by line
	
	vocabulary = data_util.get_vocabulary(keyword_poetrys)
	print("vocabulary = ",vocabulary[0:10])

	# fpx = open(temp + 'keyword_poetrys.pkl','wb')
	# pickle.dump(keyword_poetrys,fpx)
	# fpx.close()

	# fpx = open(temp + 'keyword_poetrys.pkl','rb')
	# keyword_poetrys = pickle.load(fpx)
	# # keyword_poetrys = polish_poem(keyword_poetrys)
	# fpx.close()
	# print("keyword_poetrys = ",len(keyword_poetrys))
	# print("keyword_poetrys = ",keyword_poetrys[0:3])

	xdata,ydata = data_util.poem_to_vector(keyword_poetrys,vocabulary)
	print("xdata = ",xdata.shape)
	print("ydata = ",ydata.shape)
	np.save("../data/xdata",xdata)
	np.save("../data/ydata",ydata)