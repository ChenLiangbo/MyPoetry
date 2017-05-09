#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import os

target = '../newdata/quatrain7'

def delete_5words(filename,outdir):
	out = filename.split('/')[-1]
	fp = open(filename, "r", encoding='utf-8',)
	writer = open(target,'a',encoding='utf-8')
	for line in fp:
		line = line.strip('\n').replace(' ','').replace('\t',',')
		if len(line.split(',')[0]) > 5  and 'R' not in line :
			# print("line = ",line)
			writer.write(line)
			writer.write('\n')
			# break
	writer.close()


if __name__ == '__main__':
	filedir = '../dataset/'
	outdir = '../newdata/'
	filelist = os.listdir(filedir)
	print("filelist = ",filelist)
	for f in filelist:
		filename = filedir + f
		# print("filename = ",filename.split('/')[-1])
		delete_5words(filename,outdir)
		# break

