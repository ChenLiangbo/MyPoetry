#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import data_util
import numpy as np

if __name__ == '__main__':
	import pickle
	dataset = '../newdata/'
	temp = '../data/'
	poetry_file = dataset + 'quatrain7'
	origin_poetrys = data_util.read_data(poetry_file)
	vocabulary = data_util.get_vocabulary(origin_poetrys)
	word_num_map = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
	num_word_map = dict(zip(range(len(vocabulary)),vocabulary))
	print("vocabulary = ",len(vocabulary),vocabulary[0:10])

	keyword_poetrys = keyword_poem(poetry_file)
	keyword_poetrys = polish_poem(keyword_poetrys)
	keyword_poetrys = data_util.keyword_line(poetry_file)
	fpx = open(temp + 'keyword_poetrys.pkl','wb')
	pickle.dump(keyword_poetrys,fpx)
	fpx.close()

	fpx = open(temp + 'keyword_poetrys.pkl','rb')
	keyword_poetrys = pickle.load(fpx)
	# keyword_poetrys = polish_poem(keyword_poetrys)
	fpx.close()
	print("keyword_poetrys = ",len(keyword_poetrys))
	
	xdata,ydata = data_util.poem_to_vector(keyword_poetrys,vocabulary)
	print("xdata = ",xdata.shape)
	print("ydata = ",ydata.shape)
	np.save("../data/xdata",xdata)
	np.save("../data/ydata",ydata)

	# one_hot_x,one_hot_y = data_util.one_hot_vectorize(keyword_poetrys,vocabulary)
	# print("one_hot_x = ",one_hot_x.shape)
	# print("one_hot_y = ",one_hot_y.shape)
	# np.save("../data/one_hot_x",one_hot_x)
	# np.save("../data/one_hot_y",one_hot_y)

