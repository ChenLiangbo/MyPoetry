#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np


# inputs:列表，列表的每一个元素表示一个样本序列
# inputs_time_major  矩阵，每一列表示一个样本
# sequence_lengths 列表，每一个元素为inputs_time_major 每一列对应的非零元素个数
def batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

# 随机生成一串在给定范围内的数字作为一个列表，将许多个列表组合为迭代器的一个迭代返回
# 迭代器的每一个迭代都是列表，列表中的每一个元素表示一个样本，每个样本的长度随机
def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]