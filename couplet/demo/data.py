#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import os
import numpy as np
import collections
import pickle

EOS = 1  # 'EOS' in vocabulary
PAD = 0
# 从整理好的对联集里面读出对联数据
# 返回列表 列表组成为[[上联，下联],[上联，下联]]
# 上联结尾标识符中文；（分号），下联结尾标识符中文。（句号）
def read_couplet(filename):
	couplet = []  
	with open(filename,'r',encoding = 'utf-8') as f:
		for line in f:
			if ',' in line:
				line = line.replace(',','，')
			if '.' in line:
				line = line.replace('.','。')
			if ';' in line:
				line = line.replace(';','；')

			line = line.strip('\n')
			couplet.append(line.split(' '))
	return couplet

# 统计对联数据集中出现的所有汉字或者标点符号
def get_vacobulary(couplet):
	line_list = []
	for i in range(len(couplet)):
		line_list.append(couplet[i][0])
		line_list.append(couplet[i][1])

	# 统计每个字出现次数
	all_words = ''.join(line_list)

	counter = collections.Counter(all_words)
	count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 根据出现次数的多少进行排序
	words, number = zip(*count_pairs)
	
	vocabulary = (' ','EOS') + words[:len(words)]
	# vocabulary = words
	return vocabulary

# 将词库写入文件
def write_vocabulary(vocabulary,ddir):
	fp = open(ddir + 'vocabulary','w')
	for v in vocabulary:
		fp.write(v + '\n')
	fp.close()

# 读取写入文件的词库
def read_vocabulary(ddir):
	vocabulary = []
	with open(ddir + 'vocabulary','r') as f:
		for line in f:
			vocabulary.append(line.strip('\n'))
	return vocabulary

# 所有对联的最大长度
def get_length(couplet):
	max_length = 0
	for c in couplet:
		if len(c[0]) > max_length:
			max_length = len(c[0])
	return max_length

# 一句中文对联转换为一个数值化的向量，返回值是列表
# 每个向量都以终止符EOS结尾
def encode_to_vector(astring,word_to_num):
	vector = []
	for w in astring:
		if w == 'EOS':
			break
		vector.append(word_to_num[w])
	vector.append(EOS)
	return vector

# 将一个向量表示的句子转换成汉字表示的字符串，返回值是字符串
# 遇到终止符EOS = 1 就退出
def decode_to_string(vector,num_to_word):
	try:
		vector = vector.tolist()
	except Exception as ex:
		print("[Exception Information] ",ex)
	astring = ''
	for v in vector:
		if v == EOS:
			break
		astring = astring + num_to_word[v]
	return astring


def couplet_to_vector(couplet,vocabulary):
	word_to_num = dict(zip(vocabulary,range(len(vocabulary))))
	num_to_word = dict(zip(range(len(vocabulary)),vocabulary))
	max_length = get_length(couplet) + 1  # 需要加上终止符
	xdata = []
	ydata = []
	for c in couplet:
		x = [0]*max_length
		couplet_x = c[0]
		couplet_xn = encode_to_vector(couplet_x,word_to_num)
		x[:len(couplet_xn)] = couplet_xn
		xdata.append(x)
		y = [0]*max_length
		couplet_y = c[1]
		couplet_yn = encode_to_vector(couplet_y,word_to_num)
		y[:len(couplet_yn)] = couplet_yn
		ydata.append(y)
	return np.array(xdata),np.array(ydata)

# 根据比例将数据集拆分为 training set,validation set,test set
# dsplit = [9.5,0.2,0.3]
def split_data(xdata,ydata,dsplit = [9.5,0.2,0.3]):
	shape = xdata.shape
	length = int(dsplit[0]*shape[0]/10)
	length1= int(dsplit[1]*shape[0]/10)
	trainX = xdata[:length,:]
	trainY = ydata[:length,:]
	validX = xdata[length:length+length1,:]
	validY = ydata[length:length+length1,:]
	testX  = xdata[length+length1:,:]
	testY  = ydata[length+length1:,:]
	return trainX,trainY,validX,validY,testX,testY


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
	data = '../data/'
	if not os.path.exists(data):
		os.mkdir(data)

	filename = '../dataset/couplet.txt'
	couplet = read_couplet(filename)
	print("couplet = ",len(couplet))
	vocabulary = get_vacobulary(couplet)
	print("vocabulary = ",len(vocabulary),vocabulary[0:10])
	write_vocabulary(vocabulary,data)
	vocabulary = read_vocabulary(data)
	print("vocabulary = ",len(vocabulary),vocabulary[0:10])
	word_to_num = dict(zip(vocabulary,range(len(vocabulary))))
	num_to_word = dict(zip(range(len(vocabulary)),vocabulary))

	xdata,ydata = couplet_to_vector(couplet,vocabulary)
	print("xdata = ",xdata.shape)
	print("ydata = ",ydata.shape)
	np.save(data + 'xdata',xdata)
	np.save(data + 'ydata',ydata)
	print("-"*80)
	print("encode-decode example")
	rnum = 5000
	vector_up  = xdata[rnum]
	decoded_up = decode_to_string(vector_up,num_to_word)
	encoded_up = encode_to_vector(decoded_up,word_to_num)
	print("vector_up = ",vector_up)
	print("decoded_up = ",decoded_up)
	print("encoded_up = ",encoded_up)

	vector_down  = ydata[rnum]
	decoded_down = decode_to_string(vector_down,num_to_word)
	encoded_down = encode_to_vector(decoded_down,word_to_num)
	print("vector_down = ",vector_down)
	print("decoded_down = ",decoded_down)
	print("encoded_down = ",encoded_down)