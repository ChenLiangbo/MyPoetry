import tensorflow as tf
import numpy as np
import datetime
import sys
import os

# https://github.com/suriyadeepan/practical_seq2seq
# http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/


class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len, 
            xvocab_size, yvocab_size,
            emb_dim, num_layers, ckpt_path,
            lr=0.0001, 
            epochs=10001, model_name='CoupletModel'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name

        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)

        # build thy graph
        #  attach any part of the graph that needs to be exposed, to the self
        def __graph__():

            # placeholders
            tf.reset_default_graph()
            #  encoder inputs : list of indices of length xseq_len
            self.enc_ip = [ tf.placeholder(shape=[None,], 
                            dtype=tf.int64, 
                            name='ei_{}'.format(t)) for t in range(xseq_len) ]

            #  labels that represent the real outputs
            self.labels = [ tf.placeholder(shape=[None,], 
                            dtype=tf.int64, 
                            name='ei_{}'.format(t)) for t in range(yseq_len) ]

            #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
            self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]


            # Basic LSTM cell wrapped in Dropout Wrapper
            self.keep_prob = tf.placeholder(tf.float32)
            # define the basic cell
            # basic_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
            #         tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(emb_dim, state_is_tuple=True),
            #         output_keep_prob=self.keep_prob)
            basic_cell = tf.contrib.rnn.rnn_cell.DropoutWrapper(
                    tf.contrib.rnn.rnn_cell.BasicLSTMCell(emb_dim, state_is_tuple=True),
                    output_keep_prob=self.keep_prob)  # tf 0.12.0

            # stack cells together : n layered model
            # stacked_lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)
            stacked_lstm = tf.contrib.rnn.rnn_cell.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True) # tf 0.12.0


            with tf.variable_scope('decoder') as scope:
                # build the seq2seq model 
                #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions

                # self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip, stacked_lstm,
                #                                     xvocab_size, yvocab_size, emb_dim)
          

                # atteintion based seq2seq
                self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_attention_seq2seq(
                    self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                    feed_previous=False)

                # share parameters
                scope.reuse_variables()

                # self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
                #     self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                #     feed_previous=True)

                # atteintion based seq2seq
                self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_attention_seq2seq(
                    self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                    feed_previous=True)

            # now, for training,
            #  build loss function

            # weighted loss
            #  TODO : add parameter hint
            loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
            # self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size)
            self.loss = tf.nn.seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size) # tf 0.12.0
            # train op to minimize the loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        # sys.stdout.write('<log> Building Graph ')
        print("Building Graph successfully!")
        # build comput graph
        __graph__()
        # sys.stdout.write('</log>')
        
        
    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        return feed_dict

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v

    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

    # finally the train function that
    #  runs the train_op in a session
    #   evaluates on valid set periodically
    #    prints statistics
    def train(self, train_set, valid_set, sess=None ):
        # we need to save the model periodically
        saver = tf.train.Saver()
        # if no session is given
        if not sess:
            try:
                sess = self.restore_last_session()
                # sess = self.load_model()
                print("Load model okay, train again!")
            except Exception as ex:
                print("[Exception Information] ",str(ex))
                # create a session
                sess = tf.Session()
                # init all variables
        sess.run(tf.global_variables_initializer())

        # sys.stdout.write('\n<log> Training started </log>\n')
        print("Training started ...")
        # run M epochs
        for i in range(self.epochs):
            try:
                if i % 100 == 0:
                    # print("i = ",i)
                    val_loss = self.eval_batches(sess, valid_set, 32)
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print('epoch = %d,time = %s, val_loss = %f ' % (i,now,val_loss))

                self.train_batch(sess, train_set)
                if i % 1000 == 0: 
                    # saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    saver.save(sess, self.ckpt_path + self.model_name + '.ckpt')
                    # evaluate to get validation loss
                    # val_loss = self.eval_batches(sess, valid_set, 16) # TODO : and this
                    print('\nModel saved to disk at iteration #{}'.format(i))
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print("time = %s" % (now,))
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                saver.save(sess, self.ckpt_path + self.model_name + '.ckpt')
                return sess
        saver.save(sess, self.ckpt_path + self.model_name + '.ckpt')
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("time = %s" % (now,))

    def restore_last_session(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        print("load model okay!")
        return sess



    def load_model(self,):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess,self.ckpt_path + self.model_name + '.ckpt')
        print("load model okay!")
        return sess

    # prediction
    def predict(self, sess, X):
        feed_dict = {self.enc_ip[t]: X[:,t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        print("dec_op_v = ",len(dec_op_v),len(dec_op_v[0]))
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        print("dec_op_v = ",dec_op_v.shape)
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)


    def predict_one(self, sess, x):
        feed_dict = {self.enc_ip[t]: x[:,t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)



