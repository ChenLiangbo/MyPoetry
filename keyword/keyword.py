#!usr/bin/env/python 
# -*- coding: utf-8 -*-

# TextRank4h
'''
import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence


file = './data/yinghua.txt'
text = codecs.open(file,'r','utf-8').read()
word = TextRank4Keyword()
word.analyze(text,window = 2,lower = True)
w_list = word.get_keywords(num = 20,word_min_len = 1)
print('关键词:')
for w in w_list:
	print("word = %s,weight = %f "% (w.word,w.weight))
phrase=word.get_keyphrases(keywords_num=5,min_occur_num=2) 
print("关键词组")
for p in phrase:
	print("p = ",p)
sentence=TextRank4Sentence()  
sentence.analyze(text,lower = True)
s_list = sentence.get_key_sentences(num = 3,sentence_min_len = 5)  
print("关键句子：")
for s in s_list:
	print("s.sentence = %s,s.weight = %f " %(s.sentence,s.weight))
print("finished ...")
'''


# rake
import rake
import operator

# 每个词至少有5个字符,每个短语至少有3个词,每个关键词至少在文本中出现4次
# rake_object = rake.Rake("SmartStoplist.txt", 5, 3, 4)

stoppath = "SmartStoplist.txt"
rake_object = rake.Rake(stoppath)

text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility " \
       "of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. " \
       "Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating"\
       " sets of solutions for all types of systems are given. These criteria and the corresponding algorithms " \
       "for constructing a minimal supporting set of solutions can be used in solving all the considered types of " \
       "systems and systems of mixed types."

sentenceList = rake.split_sentences(text)
stopwordpattern = rake.build_stop_word_regex(stoppath)
phraseList = rake.generate_candidate_keywords(sentenceList, stopwordpattern)

wordscores = rake.calculate_word_scores(phraseList)
keywordcandidates = rake.generate_candidate_keyword_scores(phraseList, wordscores)
sortedKeywords = sorted(keywordcandidates.iteritems(), key=operator.itemgetter(1), reverse=True)
totalKeywords = len(sortedKeywords)
 
for keyword in sortedKeywords[0:(totalKeywords / 3)]:
    print "Keyword: ", keyword[0], ", score: ", keyword[1]