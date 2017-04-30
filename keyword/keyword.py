#!usr/bin/env/python 
# -*- coding: utf-8 -*-

# TextRank4h

import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence


file = './data/yinghua.txt'
text = codecs.open(file,'r','utf-8').read()
print("text = ",type(text))
word = TextRank4Keyword()
word.analyze(text,window = 2,lower = True)
w_list = word.get_keywords(num = 20,word_min_len = 1)
print("-"*20 + "关键词" + '-'*20 )
for w in w_list:
	print("word = %s,weight = %f "% (w.word,w.weight))
phrase=word.get_keyphrases(keywords_num=5,min_occur_num=2) 
print("-"*40)

for p in phrase:
	print("p = ",p)
print("-"*20 + "关键词组" + '-'*20 )
sentence=TextRank4Sentence()  
sentence.analyze(text,lower = True)
s_list = sentence.get_key_sentences(num = 3,sentence_min_len = 5)  
print("-"*20 + "关键句子" + '-'*20 )
for s in s_list:
	print("sentence = %s,weight = %f " %(s.sentence,s.weight))
print("finished ...")


