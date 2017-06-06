#!usr/bin/env/python3
#coding:utf-8
import tornado
import tornado.httpclient
import json

url = 'http://127.0.0.1:7777/couplet/'
data = {"coupletUp":"西出阳关多故人"}


body = json.dumps(data)
http_client = tornado.httpclient.AsyncHTTPClient()

req = tornado.gen.Task(
    http_client.fetch, 
    url,
    method="POST", 
    # headers=headers,
    body=body, 
    validate_cert=False)

def handler_response(response):
    # print("response = ",dir(response))
    print("body = ",response.body)
    print("-"*80)
for i in range(10):
    http_client.fetch(req, handler_response)
tornado.ioloop.IOLoop.instance().start()