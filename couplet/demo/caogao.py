#!usr/bin/env/python 
# -*- coding: utf-8 -*-


website = 'http://www.cnblogs.com/txw1958/'
print("website = ",type(website))   # str
website_bytes_utf8 = website.encode(encoding="utf-8")
print("website_bytes_utf8 = ",type(website_bytes_utf8))  # bytes

website_bytes_gb2312 = website.encode(encoding="gb2312")  # bytes