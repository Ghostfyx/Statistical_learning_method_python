#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/2 13:40
# @Author  : fanyuexiang
# @Site    : 
# @File    : DecisonTree.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from math import log
import pprint


def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels
datasets, labels = create_data()
df = pd.DataFrame(datasets, columns=labels)
df = df.replace({'年龄': {'青年': 1, '中年': 2, '老年': 3}, '有工作': {'是': 1, '否': 0},
                 '有自己的房子': {'是': 1, '否': 0}, '信贷情况': {'一般': '1', '好': 2, '非常好': '3'},
                 '类别': {'是': 1, '否': 0}})
m, n = df.shape

def calc_entropy(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(p/data_length)*log(p/data_length, math.e) for p in label_count.values()])
    return ent
# 经验条件熵
def cond_ent(datasets, axis=0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    cond_ent = sum([(len(p)/data_length)*calc_entropy(p) for p in feature_sets.values()])
    print(cond_ent)
    return cond_ent
cond_ent(datasets)