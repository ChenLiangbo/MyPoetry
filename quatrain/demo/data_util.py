#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import collections
import numpy as np
from textrank4zh import TextRank4Keyword
import random 

# 由一首诗 使用TextRank算法提取关键词，生成 [关键词，前文，当前行]组成的数据
# ['梅', '', '偶寻半开梅']
# ['闲', '偶寻半开梅', '闲倚一竿竹']
# ['儿童', '偶寻半开梅,闲倚一竿竹', '儿童不知春']
# ['问草', '偶寻半开梅,闲倚一竿竹,儿童不知春', '问草何故绿']
def keyword_poem(poetry_file):
    poetrys = []
    ranker = TextRank4Keyword()
    j = 0
    with open(poetry_file, "r", encoding='utf-8',) as f:
    	for line in f:
    		line = line.strip('\n')
    		# print("line = ",line)
    		lines = line.split(',')
    		# print("lines = ",lines)
    		ranker.analyze(line,window = 2,lower = True)
    		w_list = ranker.get_keywords(num = 8,word_min_len = 1)
    		length = len(lines)
    		for i in range(length):
    			xline = []  # [keyword,context,line]
    			for w in w_list:
    				# print("w = ",w.word)
    				if len(w.word) > 3:
    					continue
    				if w.word in lines[i]:
    					xline.append(w.word)
    					xline.append(','.join(lines[0:i]))
    					xline.append(lines[i])
    					# print("xline = ",xline)
    					poetrys.append(xline)
    					break
    		j = j + 1
    		if j % 100 == 0:
    			print("j = ",j)
    			# break
    		# break
    # print("poetrys = ",poetrys)
    return poetrys

# ['负才', '道蕴谈锋不落诠,耳根何福受清圆,自知语乏烟霞气', '枉负才名三十年']
# use textRank get keyword for every line
def keyword_line(poetry_file):
    poetrys = []
    ranker = TextRank4Keyword()
    j = 0
    with open(poetry_file, "r", encoding='utf-8',) as f:
    	for line in f:
    		line = line.strip('\n')  # a line is a poem
    		# print("line = ",line)
    		lines = line.split(',')
    		for i in range(len(lines)):
    			xline = []
    			ranker.analyze(lines[i],window = 2,lower = True)
    			w_list = ranker.get_keywords(num = 3,word_min_len = 1)
    			for w in w_list:
    				if len(w.word) <= 3:
    					xline.append(w.word)
    					xline.append(','.join(lines[:i]))
    					xline.append(lines[i])
    					poetrys.append(xline)
    					# print("xline = ",xline)
    					break

    		j = j + 1
    		if j % 200 == 0:
    			print("j = ",j)
    			break
    		# break
    # print("poetrys = ",poetrys)
    return poetrys


# 根据原始的诗词生成词库，将词库按照词出现的频率排列，频率大的靠前
# [keyword,context,line]
def get_vocabulary(keyword_poetrys):
	poetrys = []
	for i in range(len(keyword_poetrys)):
		poetrys.append(keyword_poetrys[i][2])
	# print("poetrys = ",poetrys[1])

	# 统计每个字出现次数
	all_words = ','.join(poetrys)

	counter = collections.Counter(all_words)
	count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 根据出现次数的多少进行排序
	words, number = zip(*count_pairs)
	
	vocabulary = (' ',) + words[:len(words)]
	# vocabulary = words
	return vocabulary

# get max length of poem,[keyword,pre-context,line]
# 确定每个部分的最大长度，关键词，前文，当前行
def max_keyword(keyword_poetrys):
	# print('+'*80)
	# print(keyword_poetrys[0])
	max_list = [0,0,0]
	for kp in keyword_poetrys:
		if len(kp[0]) > max_list[0]:
			max_list[0] = len(kp[0])
		if len(kp[1]) > max_list[1]:
			max_list[1] = len(kp[1])
		if len(kp[2]) > max_list[2]:
			max_list[2] = len(kp[2])
			# print("kp2 = ",kp)
	# print('+'*80)
	return max_list

# a string line poem to number list
def string_to_num(aline,word_num_map):
	if len(aline) == 0:
		return [0,]
	ret = []
	for s in aline:
		try:
			num = word_num_map[s]
		except Exception as ex:
			print("[Exception Information]",str(ex))
			num = 0

		ret.append(num)
	return ret

# keyword_max = 7,23,7
# x[0:7]  = keyword
# x[7:23] = previous context
# y[0:7]  = line

def poem_to_vector(keyword_poetrys,vocabulary):
	max_list = max_keyword(keyword_poetrys)
	print("max_list = ",max_list)
	word_num_map = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
	xdata = []
	ydata = []
	for kp in keyword_poetrys:
		# print("kp = ",kp)
		x1 = [0]*max_list[0]
		n1 = string_to_num(kp[0],word_num_map)
		x1[:len(n1)] = n1
		x2 = [0]*max_list[1]
		n2 = string_to_num(kp[1],word_num_map)
		x2[:len(n2)] = n2
		x1.extend(x2)
		xdata.append(x1)

		y  = [0]*(max_list[0] + max_list[1])  # x y same length 
		# y  = [0]*max_list[2]  # x y different length
		yn = string_to_num(kp[2],word_num_map)
		y[:len(yn)] = yn
		ydata.append(y)

		# break
	return np.array(xdata),np.array(ydata)

# x = (None,30)
# y = (None,7,voca_size)
def one_hot_vectorize(keyword_poetrys,vocabulary):
	max_list = max_keyword(keyword_poetrys)
	word_num_map = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
	# num_word_map = dict(zip(range(len(vocabulary)),vocabulary))
	voca_zie = len(vocabulary)
	x = []
	y = []
	length = len(keyword_poetrys[0][-1])
	print("length = ",length)
	for kp in keyword_poetrys:
		x1 = [0]*max_list[0]
		n1 = string_to_num(kp[0],word_num_map)
		x1[:len(n1)] = n1
		x2 = [0]*max_list[1]
		n2 = string_to_num(kp[1],word_num_map)
		x2[:len(n2)] = n2
		x1.extend(x2)
		y1 = np.zeros((length,voca_zie))
		for i in range(length):
			word = kp[2][i]
			index = word_num_map[word]
			y1[i:index] = 1
		y.append(y1)
	return np.array(x),np.array(y)


def print_poem(poem):
	print("-"*30 + 'poem' + '-'*30)
	for line in poem:
		print(line)
	print("-"*30 + 'poem' + '-'*30)


# 由一个关键词，前文生成一个向量,返回数据列表
# max_list = [7,23,7]
def get_xsample(keyword,context,max_list,word_num_map):
	x1 = [0]*max_list[0]
	n1 = string_to_num(keyword,word_num_map)
	x1[:len(n1)] = n1
	x2 = [0]*max_list[1]
	n2 = string_to_num(context,word_num_map)
	x2[:len(n2)] = n2
	x1.extend(x2)
	return x1

# 给定一段文章 提取不超过四个关键词
def get_4keyword(context):
	ranker = TextRank4Keyword()
	ranker.analyze(text,window = 2,lower = True)
	w_list = ranker.get_keywords(num = 20,word_min_len = 1)
	keyword_list = []
	i = 0
	for w in w_list:
		keyword_list.append(w.word)
		i = i + 1
		if i > 4:
			break
	return keyword_list


# transform a digit to poem 
def num_to_poem(seq_y,vocabulary):
	num_word_map = dict(zip(range(len(vocabulary)),vocabulary))
	poem = []
	seq_y = seq_y.tolist()
	for y in seq_y[0:7]:
		poem.append(num_word_map[y])
	return poem


def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = random.sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T


def batch_gen(x, y, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T


if __name__ == '__main__':
	import pickle
	dataset = '../newdata/'
	temp = '../data/'
	poetry_file = dataset + 'quatrain7'

	# keyword_poetrys = keyword_poem(poetry_file) # get keyword from whole poem
	# keyword_poetrys = keyword_line(poetry_file) # get keyword line by line


	# fpx = open(temp + 'keyword_poetrys.pkl','wb')
	# pickle.dump(keyword_poetrys,fpx)
	# fpx.close()

	fpx = open(temp + 'keyword_poetrys.pkl','rb')
	keyword_poetrys = pickle.load(fpx)
	# keyword_poetrys = polish_poem(keyword_poetrys)
	fpx.close()
	print("keyword_poetrys = ",len(keyword_poetrys))
	# print("keyword_poetrys = ",keyword_poetrys[0:3])

	vocabulary = get_vocabulary(keyword_poetrys)
	print("vocabulary = ",len(vocabulary))

	xdata,ydata = poem_to_vector(keyword_poetrys,vocabulary)
	print("xdata = ",xdata.shape)
	print("ydata = ",ydata.shape)
	np.save("../data/xdata",xdata)
	np.save("../data/ydata",ydata)

	# one_hot_x,one_hot_y = data_util.one_hot_vectorize(keyword_poetrys,vocabulary)
	# print("one_hot_x = ",one_hot_x.shape)
	# print("one_hot_y = ",one_hot_y.shape)
	# np.save("../data/one_hot_x",one_hot_x)
	# np.save("../data/one_hot_y",one_hot_y)


