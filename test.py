#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:54:25 2017

@author: brian
"""

from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import numpy as np
import random

# 读取数据集
mem=Memory('./mycache1')
@mem.cache
def get_data(filename):
    data=load_svmlight_file(filename, n_features=123)
    return data[0], data[1]

# 划分数据集
x_train, y_train = get_data('../data/a9a')
x_test, y_test = get_data('../data/a9a.t')

x_train = x_train.toarray()
x_test = x_test.toarray()

x_train=np.c_[ x_train, np.ones(x_train.shape[0]) ]     # 在样本数据最后添加一整列的1，代替参数b
x_test=np.c_[ x_test, np.ones(x_test.shape[0]) ]     # 在样本数据最后添加一整列的1，代替参数b

# 求梯度的函数
def grad(times, w, theLambda):
    gradw = np.zeros(124)
    for i in range(0, times):
        index = random.randint(0, x_train.shape[0] - 1)
        tmp = np.exp( -1 * y_train[index] * np.dot(w, x_train[index]) )
        gradw += -1 * y_train[index] * x_train[index] * tmp
    return ( theLambda * w * w + gradw ) / ( (1+tmp) * times )

# 求准确率
def assess(w):
    right = 0.
    for i in range(0, x_test.shape[0]):
        if 1 / ( 1 + np.exp(-1 * np.dot(w, x_test[i])) ) >= 0.5:
            if y_test[i] == +1:
                right += 1
        else:
            if y_test[i] == -1:
                right += 1
    return right / x_test.shape[0]

# 计算loss
def getLoss(w):
    label_validate = []
    for i in range(0, x_test.shape[0]):
        if 1 / ( 1 + np.exp(-1 * np.dot(w, x_test[i])) ) >= 0.5:
            label_validate.append(1.)
        else:
            label_validate.append(-1.)
    cur=0.
    for i in range(0, x_test.shape[0]):
        cur += max(0, 1. - y_test[i] * label_validate[i])
    return cur / x_test.shape[0]

# AdaDelta参数初始化
w_adadelta = np.random.rand(124)
capital_g_adadelta = np.random.rand(124) 
gama_adadelta = 0.95
delta_t = np.random.rand(124)
lambda_adadelta = 1.
epsilon = 1e-8

ranges = range(0, 300)
loss_adadelta = []
accuracy_adadelta = []

for i in ranges:
    gradw = grad(2**4, w_adadelta, lambda_adadelta)     # 提高样本数目，减少震荡
    capital_g_adadelta = gama_adadelta * capital_g_adadelta + (1-gama_adadelta) * gradw * gradw
    delta_w = -1 * ( np.sqrt( delta_t + epsilon ) / ( np.sqrt( capital_g_adadelta + epsilon ) ) ) * gradw
    w_adadelta += delta_w
    delta_t = gama_adadelta * delta_t + (1-gama_adadelta) * delta_w * delta_w
    
    loss = getLoss(w_adadelta)
    loss_adadelta.append(loss)
    accuracy = assess(w_adadelta)
    accuracy_adadelta.append(accuracy)
    
print('计算结束')