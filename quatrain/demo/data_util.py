#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import collections
import numpy as np
from textrank4zh import TextRank4Keyword

# 数据预处理
# 诗集 ['偶寻半开梅,闲倚一竿竹,儿童不知春,问草何故绿','','']
def read_data(poetry_file):
    poetrys = []
    with open(poetry_file, "r", encoding='utf-8',) as f:
    	for line in f:
    		try:
    			poem = []
    			line = line.strip('\n')
    			line = line.split('\t')
    			poem = poem + line[0].split(' ') + [','] + line[1].split(' ') + ['.'] +  line[2].split(' ') + [','] + line[3].split(' ') + ['.']
	    		poetrys.append(poem)
	    	except Exception as ex:
	    		print("[Exception Information]",str(ex))
	    		pass
    return poetrys

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
    		line = line.replace('\t',',').replace(' ','')
    		lines = line.split(',')
    		ranker.analyze(line,window = 2,lower = True)
    		w_list = ranker.get_keywords(num = 8,word_min_len = 1)
    		length = len(lines)
    		for i in range(length):
    			for w in w_list:
    				xline = []  # [keyword,context,line]
    				if w.word in lines[i]:
    					xline.append(w.word)
    					xline.append(','.join(lines[0:i]))
    					xline.append(lines[i])
    					# print("xline = ",xline)
    					poetrys.append(xline)
    					break
    				# print("xline = ",xline)
    		j = j + 1
    		if j % 100 == 0:
    			print("j = ",j)
    			break

    		# break
    return poetrys

# 数据集的一些诗句含有不确定汉字'R','俯观<R><R>总尘劳'
def polish_poem(keyword_poetrys):
	poetrys = []
	for kp in keyword_poetrys:
		if ('R' not in kp[0]) and ('R' not in kp[1]) and ('R' not in kp[2]):
			poetrys.append(kp)
	return poetrys

def get_vocabulary(origin_poetrys):
	poetrys = sorted(origin_poetrys,key=lambda line: len(line)) # 按诗的字数排序
	# 统计每个字出现次数
	all_words = []
	for poetry in poetrys:
		all_words += [word for word in poetry]

	counter = collections.Counter(all_words)
	count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 根据出现次数的多少进行排序
	words, number = zip(*count_pairs)
	# print("words = ",len(words),words[0:10])  # ('.', ',', '不', '人')

	vocabulary = ('',) + words[:len(words)]
	word_num_map = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
	return vocabulary,word_num_map


def max_keyword(keyword_poetrys):
	max_list = [0,0,0]
	for kp in keyword_poetrys:
		if len(kp[0]) > max_list[0]:
			max_list[0] = len(kp[0])
		if len(kp[1]) > max_list[1]:
			max_list[1] = len(kp[1])
		if len(kp[2]) > max_list[2]:
			max_list[2] = len(kp[2])
			print("kp2 = ",kp)
	return max_list


def string_to_num(aline,word_num_map):
	if len(aline) == 0:
		return [0,]
	ret = []
	for s in aline:
		ret.append(word_num_map[s])
	return ret

# keyword_max = 7,23,7
# x[0:7]  = keyword
# x[7:23] = previous context
# y[0:7]  = line

def poem_to_vector(keyword_poetrys,vocabulary):
	keyword_max = max_keyword(keyword_poetrys)
	word_num_map = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
	xdata = []
	ydata = []
	for kp in keyword_poetrys:
		print("kp = ",kp)
		x1 = [0]*max_list[0]
		n1 = string_to_num(kp[0])
		x1[:len(n1)] = n1
		x2 = [0]*max_list[1]
		n2 = string_to_num(kp[1])
		x2[:len(n2)] = n2
		x1.extend(x2)
		xdata.append(x1)

		y  = [0]*max_list[2] 
		yn = string_to_num(kp[2])
		y[:len(yn)] = yn
		ydata.append(y)

		break
	return np.array(xdata),np.array(ydata)

	
def to_poem(ddata,words):
	shape = ddata.shape
	# print("shape = ",shape)
	poem = []
	for i in range(shape[0]):
		line = []
		for j in range(shape[1]):
			line.append(words[ddata[i,j]])
		poem.append(line)
	return poem

def print_poem(poem):
	print("-"*30 + 'poem' + '-'*30)
	for line in poem:
		print(line)
	print("-"*30 + 'poem' + '-'*30)


if __name__ == '__main__':
	import pickle
	dataset = '../dataset/'
	temp = '../data/'
	poetry_file = dataset + 'qtrain'
	origin_poetrys = read_data(poetry_file)
	vocabulary,word_num_map = get_vocabulary(origin_poetrys)
	print("vocabulary = ",len(vocabulary),type(vocabulary))

	'''
	keyword_poetrys = keyword_poem(poetry_file)
	keyword_poetrys = polish_poem(keyword_poetrys)
	fpx = open(temp + 'keyword_poetrys.pkl','wb')
	pickle.dump(keyword_poetrys,fpx)
	fpx.close()
	'''
	fpx = open(temp + 'keyword_poetrys.pkl','rb')
	keyword_poetrys = pickle.load(fpx)
	keyword_poetrys = polish_poem(keyword_poetrys)
	fpx.close()
	print("keyword_poetrys = ",len(keyword_poetrys))
	
	poetrys_vector = poem_to_vector(keyword_poetrys,vocabulary)

	# x_batches,y_batches,words,n_chunk = get_batches(vocabulary,keyword_poetrys,batch_size = 64)
	# word_num_map = dict(zip(words, range(len(words))))  # {"人":5,""}
	# voca_size = len(words)

