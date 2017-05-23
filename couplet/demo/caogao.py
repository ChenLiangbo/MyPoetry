#!usr/bin/env/python 
# -*- coding: utf-8 -*-

def batch_gen(x, y, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T


if __name__ == '__main__':
    import numpy as np
    ddir = '../data/'
    ckpt = '../ckpt/'
    xdata = np.load(ddir + 'xdata.npy')
    ydata = np.load(ddir + 'ydata.npy')

    val_batch_gen = batch_gen(xdata,ydata,128)
    batch = next(val_batch_gen)
    x = batch[0]
    y = batch[1]
    print("x = ",x.shape,"y = ",y.shape)

