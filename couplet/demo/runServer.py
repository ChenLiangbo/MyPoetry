#!/user/bin/env/python
#coding:utf-8

import json
import datetime
import numpy as np

from model import Seq2Seq
import data as data_util

import tornado.web
from tornado import gen


ddir = '../data/'

def load_model(ddir):
    xdata = np.load(ddir + 'xdata.npy')
    ydata = np.load(ddir + 'ydata.npy')

    vocabulary = data_util.read_vocabulary(ddir)
    word_to_num = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
    num_to_word = dict(zip(range(len(vocabulary)),vocabulary))
    xseq_len = trainX.shape[-1]
    yseq_len = trainY.shape[-1]

    print("xseq_len = ",xseq_len,"yseq_len = ",yseq_len)
    # xvocab_size = len(metadata['idx2w'])
    xvocab_size = len(vocabulary)
    yvocab_size = xvocab_size
    emb_dim = 1024
    model = Seq2Seq(xseq_len = xseq_len,
        yseq_len    = yseq_len,
        xvocab_size = xvocab_size,
        yvocab_size = yvocab_size,
        ckpt_path   = ckpt,
        emb_dim     = emb_dim,
        num_layers  = 3
        )
    return model,vocabulary





class BaseHandler(tornado.web.RequestHandler):

    # db_T0tables = dbOperator.T0tablesOperator
    def setGetRequestHeader(self):
    	'''deal with cross domain trouble '''
        self.set_header('Access-Control-Allow-Origin','*')
        self.set_header('Access-Control-Allow-Methods',self.request.method)
        self.set_header('Access-Control-Allow-Headers',"x-requested-with,content-type")    

    def setPostRequestHeader(self):
    	'''deal with cross domain trouble '''
        self.set_header('Access-Control-Allow-Origin','*')
        self.set_header('Access-Control-Allow-Methods',self.request.method)
        self.set_header('Access-Control-Allow-Headers',"x-requested-with,content-type")
    


'''登录验证  使用post方法，接受username和password参数'''
class CoupletHandler(BaseHandler):
    def get(self):
        self.setGetRequestHeader()                	 
     	data = {"coupletUp":"chenlb","coupletDown":"chenliangbohk1688"}
    	data = json.dumps(data,separators = (',',':'))  	
        print("----------------------------------------")
        self.write(data)

    def post(self):
        print("---------------------------running in CoupletHandler post--------------------------------")
        data = {}
        self.setPostRequestHeader() 
        try:
            bodyString = self.request.body
            bodyDict = json.loads(bodyString)
            coupletUp = bodyDict["coupletUp"]
            print("coupletUp = ",coupletUp)
        except Exception as ex:
            print("Exception happens when get data in CoupletHandler post:",ex)
            self.write(data)
            return

        model,vocabulary = load_model(ddir)
        word_to_num = dict(zip(vocabulary, range(len(vocabulary)))) # {".":0,",":1,"不":2,"人":3}
        num_to_word = dict(zip(range(len(vocabulary)),vocabulary))

        couplet_x = data_util.encode_to_vector(coupletUp,word_to_num)
        
        sess = model.restore_last_session()
        couplet_y = model.predict_one(sess,couplet_x)
        coupletDown = data_util.decode_to_string(couplet_y,num_to_word)
        data = {"message":'ok',"code":200,"coupletDown":coupletDown}

        data = json.dumps(data,separators = (',',':'))  
        self.write(data)


import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options

define("port", default=7777, help="run on the given port", type=int)

app = tornado.web.Application(
    handlers=[(r'/couplet/', CoupletHandler),
    ],           
    debug = True,
    cookie_secret = '7cbddfc12c7522bc46010a4563e80257',
    # template_path = os.path.join(os.path.dirname(__file__), "helloword2"),
    # static_path=os.path.join(os.path.dirname(__file__), "helloword2"),
     
)


def serverStart():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    serverStart()