#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
import numpy as np
import MyModel
import data_util
import pickle
import sys
from textrank4zh import TextRank4Keyword

intent = ''

keywords = data_util.get_4keyword(intent)


if len(keywords) < 4:
	print("[Warming] Get less than 4 keywords from intent")
	sys.exit()

# dataset = '../newdata/'
# poetry_file = dataset + 'quatrain7'
fpx = open(data + 'keyword_poetrys.pkl','rb')
keyword_poetrys = pickle.load(fpx)
fpx.close()
vocabulary = data_util.get_vocabulary(keyword_poetrys)
word_num_map = dict(zip(vocabulary,len(vocabulary)))

max_list = data_util.max_keyword(keyword_poetrys)

xseq_len = 3 + 23
yseq_len = 7
 
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


poem = []
for i in range(len(keywords)):
	k = keywords[i]
	if len(poem) > 0:
		context = ','.join(poem[:i])
	else:
		context = ''
	x = data_util.get_xsample(k,context,max_list,word_num_map)
	py = py= model.predict_one(sess,x)
	line = ''.join(py[0])
	poem.append(line)

data_util.print_poem(poem)


