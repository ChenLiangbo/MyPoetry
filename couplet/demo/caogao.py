#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import datetime
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    print("time is %s " % (now,))
