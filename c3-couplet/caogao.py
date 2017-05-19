#!usr/bin/env/python 
# -*- coding: utf-8 -*-

s = '今； 古。'
sset = s.split(' ')
print("sset = ",sset)
if sset[0].endswith('；'):
	print("yes")
	s0 = sset[0]
	print("s0 = ",len(s0),list(s0)[-1])