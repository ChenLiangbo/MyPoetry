#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import collections
import numpy as np
 
# 数据预处理
# 诗集 从文件中生成古诗词列表
def read_data(poetry_file):
    poetrys = []
    with open(poetry_file, "r", encoding='utf-8',) as f:
    	for line in f:
    		try:
    			poem = ['[']
    			line = line.strip('\n')
    			line = line.split('\t')
    			# print("line ---- 1 ",line)
    			poem = poem + line[0].split(' ') + [','] + line[1].split(' ') + ['.'] +  line[2].split(' ') + [','] + line[3].split(' ') + ['.']
    			poem = poem + [']',]
    			# print("poem = ",poem,len(poem))

	    		poetrys.append(poem)
	    	except Exception as ex:
	    		print("[Exception Information]",str(ex))
	    		pass
	    	# break
    return poetrys


def get_batches(poetrys,batch_size = 64):
	poetrys = sorted(poetrys,key=lambda line: len(line)) # 按诗的字数排序
	print('Number of Quatrain : ', len(poetrys))
	
	# 统计每个字出现次数
	all_words = []
	for poetry in poetrys:
		all_words += [word for word in poetry]
	counter = collections.Counter(all_words)
	count_pairs = sorted(counter.items(), key=lambda x: -x[1])
	words, _ = zip(*count_pairs)
	 
	# 取前多少个常用字
	words = words[:len(words)] + (' ',)
	
	# 每个字映射为一个数字ID
	word_num_map = dict(zip(words, range(len(words))))
	# 把诗转换为向量形式，参考TensorFlow练习1
	to_num = lambda word: word_num_map.get(word, len(words))
	poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
	
	# 每次取64首诗进行训练
	n_chunk = len(poetrys_vector) // batch_size
	x_batches = []
	y_batches = []
	for i in range(n_chunk):
		start_index = i * batch_size
		end_index = start_index + batch_size
	 
		batches = poetrys_vector[start_index:end_index]
		length = max(map(len,batches))
		xdata = np.full((batch_size,length), word_num_map[' '], np.int32)
		for row in range(batch_size):
			xdata[row,:len(batches[row])] = batches[row]
		ydata = np.copy(xdata)
		ydata[:,:-1] = xdata[:,1:]
		"""
		xdata             ydata
		[6,2,4,6,9]       [2,4,6,9,9]
		[1,4,2,8,5]       [4,2,8,5,5]
		"""
		x_batches.append(xdata)
		y_batches.append(ydata)
		# if i > 2000:
			# break
	# print("xdata = ",xdata,type(xdata))
	# print("-"*80)
	# print("ydata = ",ydata,type(ydata))
	# print('='*80)
	return x_batches,y_batches,words,n_chunk
	
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
	dataset = '../dataset/'
	poetry_file = dataset + 'qtrain'
	poetrys = read_data(poetry_file)
	x_batches,y_batches,words,n_chunk = get_batches(poetrys,batch_size = 64)
	print("y_batches = ",len(y_batches))
	print("x_batches = ",len(x_batches))
	print("words = ",len(words))
	print("n_chunk = ",n_chunk)