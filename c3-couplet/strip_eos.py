#!usr/bin/env/python 
# -*- coding: utf-8 -*-

# 去除对联最后的标点符号

def is_valid_ends(astring):
	aset = [',','.','?','!','，','；','。','？','!']
	if len(astring) == 0:
		return False
	if list(astring)[-1] in aset:
		return True
	else:
		return False

def strip_one_line(line):
	line = line.strip('\n').split(' ')
	line_up   = line[0]
	line_down = line[1]
	if is_valid_ends(line_up):
		line_up = ''.join(list(line_up)[:-1])
	if is_valid_ends(line_down):
		line_down = ''.join(list(line_down)[:-1])
	line = line_up + ' ' + line_down
	return line

def strip_file(fin_name,fout_name):
	fp_in = open(fin_name,'r',encoding = 'gbk')
	fp_out= open(fout_name,'w')
	lines = fp_in.readlines()
	for line in lines:
		line = strip_one_line(line)
		fp_out.write(line + '\n')

	fp_in.close()
	fp_out.close()

'''
In:
今； 古。
始； 终。
反； 正。
私； 公。
霜； 雪。
Out:
今 古
始 终
反 正
私 公
霜 雪
'''

if __name__ == '__main__':
	fin_name = './dataset/couplet.txt'  # 对联数据机
	fout_name= './dataset/out.txt'      # 输出文件
	strip_file(fin_name,fout_name)
