#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
r = np.random
alist = list(range(20))
print(alist)
r.shuffle(alist)
print(alist)
