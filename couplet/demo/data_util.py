#!usr/bin/env/python 
# -*- coding: utf-8 -*-


def strip_space(lines):
	for i in range(len(lines)):
		new = ''
		line = lines[i].strip('\n')
		for c in line:
			if ord(c) != 12288 and ord(c) != 32:
				new = new + c
		lines[i] = new
	return lines


def get_couplet_from_book(fin_name,fout_name):
	fp_out = open(fout_name,'w')
	fp_in = open(fin_name,'r',encoding = 'utf-8')
	lines = fp_in.readlines()
	for i in range(len(lines)):
		new = ''
		line = lines[i].strip('\n')
		for c in line:
			if ord(c) != 12288 and ord(c) != 32:
				new = new + c
		lines[i] = new
	
	start = 0
	length = 3
	end = start + length
	
	temp = 0
	while end<len(lines):
		line_up = lines[start]
		if (len(line_up) < 2) or ('【' in line_up) or (len(line_up) > 40) or ('——' in line_up) or ('：' in line_up) or ('“' in line_up):
			start = start + 1
			end = start + length
			continue
		line_down_list = lines[start+1:end]
		flag = False
	
		for i in range(len(line_down_list)):
			if len(line_up) == len(line_down_list[i]):
				flag = True
				temp = i + 2
				line_down = line_down_list[i]
				break
		if flag:
			couplet = line_up + ' ' + line_down + '\n'
			# print("couplet = ",couplet)
			try:
				fp_out.write(couplet)
			except:
				pass
			start = start + temp
		else:
			start = start + 1
		end = start + length
	fp_out.close()

def split_couplet(fin_name,):
	fp_in = open(fin_name,'r',encoding = 'utf-8')
	lines = fp_in.readlines()
	lines = strip_space(lines)
	fp_out  = open('../raw/out.txt','w')	
	fp_out1 = open('../raw/out2.txt','w')
	for line in lines:
		line_up = line
		if (len(line_up) < 2) or ('【' in line_up) or (len(line_up) > 40) or ('——' in line_up) or ('：' in line_up) or ('“' in line_up):
			continue
		if '；' not in line:
			fp_out1.write(line + '\n')
		else:
			lines = line.split('；')
			if len(lines[1]) > 2:
				couplet = lines[0] + '；' + ' ' + lines[1] + '\n'
				fp_out.write(couplet)
			else:
				fp_out1.write(line + '\n')


def dui_split(fin_name,fout_name):
	fp_in = open(fin_name,'r',encoding = 'utf-8')
	fp_out  = open(fout_name,'w')
	lines = fp_in.readlines()
	lines = strip_space(lines)
	i = 0
	for line in lines:
		line = line.strip('\n')
		# print("line = ",line,len(line))
		# for c in line:
		# 	print("c = ",c,ord(c))
		lset = line.split('；')
		if len(lset[0]) < len(lset[1]):
			couplet = lset[0] + '；' + ' ' + lset[1] + '\n'
			# print("couplet = ",couplet)
		else:
			line_up = lset[0][:len(lset[1])-1] + '；'
			line_down = lset[1]
			couplet = line_up + ' ' + line_down + '\n'
			# print("line_up = ",line_up,"line_down = ",line_down)
		fp_out.write(couplet)
		# print("length = ",lset[0],len(lset[0]),lset[1],len(lset[1]))
		# print('-'*80)

		# break


def split_c1(fin_name,fout_name):
	fp_in = open(fin_name,'r',encoding = 'utf-8')
	fp_out  = open(fout_name,'w')
	lines = fp_in.readlines()
	# lines = strip_space(lines)
	new = []
	for line in lines:
		line = line.strip('\n')
		if len(line) < 7:
			continue
		else:
			new.append(line)
	# print("new = ",len(new))
	# print(new[0:7])
	number = len(new)//7
	for i in range(0,len(new),7):
		print("i = ",i)
		line0 = new[i]
		line0 = line0.split('　')
		# print("line0 = ",line0)
		for j in range(len(line0)):
			line0[j] = line0[j].split('对')
		# print("line0 = ",line0)
		for c in line0:
			couplet = c[0] + '；' + ' ' + c[1] + '。' + '\n'
			fp_out.write(couplet)
		line1 = new[i+1]
		couplet = line1[0:5].split('对')
		couplet = couplet[0] + '；' + ' ' + couplet[1] + '。' + '\n'
		fp_out.write(couplet)
		# print("couplet = ",couplet)
		# print("couplet = ",couplet)
		couplet = line1[6:].split('对')
		couplet = couplet[0] + '；' + ' ' + couplet[1] + '。' + '\n'
		# print("couplet = ",couplet)
		fp_out.write(couplet)
		# print("couplet = ",couplet)
		line2 = new[i+2]
		# print("line2 = ",line2,len(line2))
		couplet = line2[0:3] + '；' + ' ' + line2[4:7] + '。' + '\n'
		fp_out.write(couplet)
		# print("couplet = ",couplet)
		couplet = line2[8:].split('对')
		couplet = couplet[0] + '；' + ' ' + couplet[1] + '。' + '\n'
		fp_out.write(couplet)
		# print("couplet = ",couplet)
		line3 = new[i+3].split('、')
		# print("line3 = ",line3)
		couplet = line3[0] + '；' + ' ' + line3[1] + '。' + '\n'
		fp_out.write(couplet)
		# print("couplet = ",couplet)

		line4 = new[i+4].split('、')
		# print("line4 = ",line4)
		couplet = line4[0] + '；' + ' ' + line4[1] + '。' + '\n'
		fp_out.write(couplet)
		# print("couplet = ",couplet)

		line5 = new[i+5].replace('　',',')
		line6 = new[i+6].replace('　',',')
		couplet = line5 + '；' + ' ' + line6 + '。' + '\n'
		fp_out.write(couplet)
		# # print("couplet = ",couplet)
		# # print("line5 = ",line5)		
		# # print("line6 = ",line6)


def split_c2(fin_name,fout_name):
	fp_in = open(fin_name,'r',encoding = 'utf-8')
	fp_out  = open(fout_name,'w')
	lines = fp_in.readlines()
	for line in lines:
		line = line.strip('\n')
		line = line.split('。')
		print("line = ",line)
		for l in line:
			# print("l = ",l)
			if '；' in l:
				l = l.split('；')
				couplet = l[0] + '；' + ' ' + l[1] + '。' + '\n'
				# print("couplet = ",couplet)
				fp_out.write(couplet)
			elif '对' in l:
				alist = l.split('，')
				for cj in alist:
					cj = cj.split('对')
					if len(cj) < 2:
						continue
					couplet = cj[0] + '；' + ' ' + cj[1] + '。' + '\n'
					# print("couplet = ",couplet)
					fp_out.write(couplet)
			else:
				couplet = l.split('，')
				# print("couplet = ",couplet)
				if len(couplet) < 2:
					continue
				couplet = couplet[0] + '；' + ' ' + couplet[1] + '。' + '\n'
				fp_out.write(couplet)
				

		# break

def split_c3(fin_name,fout_name):
	fp_in = open(fin_name,'r',encoding = 'utf-8')
	fp_out  = open(fout_name,'w')
	lines = fp_in.readlines()
	# lines = strip_space(lines)
	for line in lines:
		line = line.strip('\n')
		# print("line = ",line)
		line = line.split('。')
		# print("line = ",line)
		for aline in line:
			if '对' in aline:
				if aline.count('对') > 1:
					alist = aline.split('，')
					# print("alist = ",alist)
					for a in alist:
						a = a.split('对')
						if len(a) < 2:
							continue
						couplet = a[0] + '；' + ' ' + a[1] + '。' + '\n'
						# print("couplet = ",couplet)
						fp_out.write(couplet)
				elif aline.count('对') == 1:
					alist = aline.split('，')
					if len(alist)<2:
						continue
					couplet = alist[0] + '；' + ' ' + alist[1] + '。' + '\n'
					# print("couplet = ",couplet)
					fp_out.write(couplet)
					if len(alist) < 3:
						continue
					couplet = alist[2].split('对')
					couplet = couplet[0] + '；' + ' ' + couplet[1] + '。' + '\n'
					# print("couplet = ",couplet)
					fp_out.write(couplet)
				pass
			elif '；' in aline:
				alist = aline.split('；')
				couplet = alist[0] + '；' + ' ' + alist[1] + '。' + '\n'
				# print("couplet = ",couplet)
				fp_out.write(couplet)
			else:
				alist = aline.split('，')
				if len(alist) < 2:
					continue
				couplet = alist[0] + '；' + ' ' + alist[1] + '。' + '\n'
				# print("couplet = ",couplet)
				fp_out.write(couplet)


		# break

	fp_out.close()
	fp_in.close()

def split_c4(fin_name,fout_name):
	fp_in = open(fin_name,'r',encoding = 'utf-8')
	fp_out  = open(fout_name,'w')
	lines = fp_in.readlines()
	lines = strip_space(lines)
	for line in lines:
		line = line.strip('\n')
		# print("line = ",line)
		alist = line.split('。')
		# print("alist = ",alist)
		for aline in alist:
			if '；' in aline:
				aline = aline.split('；')
				couplet = aline[0] + '；' + ' ' + aline[1] + '。' + '\n'
				# print("couplet = ",couplet)
				fp_out.write(couplet)

		# break
	
	fp_out.close()
	fp_in.close()

def split_c5(fin_name,fout_name):
	fp_in = open(fin_name,'r',encoding = 'utf-8')
	fp_out  = open(fout_name,'w')
	lines = fp_in.readlines()
	for line in lines:
		if len(line) > 10:
			fp_out.write(line + '\n')
	
	fp_out.close()
	fp_in.close()

def ends_symbol(astring):
	sset = ['；','；','.','。',',','，','？','?','！','!']
	if astring.endswith('；') or astring.endswith('。'):
		return astring
	else:
		ends = astring[-1]
		if ends in sset:
			astring[-1] = ''
		else:
			return False

def f_key(x):
	return len(x)

def split_better(fin_name,fout_name):
	sset = ['；','；','.','。',',','，','？','?','！','!']
	fp_in = open(fin_name,'r',encoding = 'utf-8')
	# fp_out  = open(fout_name,'w')
	f_good = open('../raw/good.txt','w')
	f_bad  = open('../raw/bad.txt','w')
	lines = fp_in.readlines()
	lines.sort(key = f_key)
	print("lines = ",len(lines))
	for line in lines:
		line = line.strip('\n')
		if ('第' in line) and ('楼' in line):
			continue
		if ' ' in line:
			alist = line.split(' ')
		elif ' ' in line:
			alist = line.split(' ')
		else:
			try:
				alist = line.split('；')
				alist[0] = alist[0] + '；'
			except:
				f_bad.write(line)

		try:
			if len(alist[0]) == len(alist[1]):
				if alist[0][-1] in sset:
					alist[0][-1] = '；'
				else:
					alist[0] = alist[0] + '；'
				if alist[1][-1] in sset:
					alist[1][-1] = '。'
				else:
					alist[1] = alist[1] + '。'
				couplet = alist[0] + ' ' + alist[1] + '\n'
				f_good.write(couplet)
			else:
				couplet = alist[0] + ' ' + alist[1] + '\n'
				f_bad.write(couplet)
		except:
			pass

def soort_good(good_file,out_file):
	f_good = open(good_file,'r',encoding = 'utf-8')
	f_out  = open(out_file,'w')
	lines = f_good.readlines()
	lines.sort(key = f_key)
	for line in lines:
		f_out.write(line)
	f_good.close()
	f_out.close()


if __name__ == '__main__':
	fin_name = '../raw/temp.txt'
	fout_name= '../raw/out.txt'
	# get_couplet_from_book(fin_name,fout_name)
	# split_couplet(fin_name)
	# dui_split(fin_name,fout_name)
	# split_c1(fin_name,fout_name)
	# split_c2(fin_name,fout_name)
	# split_c3(fin_name,fout_name)
	# split_c4(fin_name,fout_name)
	# split_better(fin_name,fout_name)
	soort_good('../raw/good.txt','../raw/out.txt')