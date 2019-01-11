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
import graphviz


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


class TreeNode:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        # 是否为单节点树
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label:': self.label, 'feature': self.feature,'feature_name': self.feature_name, 'tree': self.tree}

    # 表示对象的可打印字符串
    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.tree[features[self.feature]].predict(features)


class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 计算墒(整个数据集维度)
    def calc_entropy(self, datasets):
        m = len(datasets)
        label_count = {}
        for i in range(m):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        calc_entropy = -sum([p / m * log(p / m, math.e) for p in label_count.values()])
        return calc_entropy

    # 经验条件熵
    def condition_entropy(self, datasets, axis=0):
        m = len(datasets)
        feature_sets = {}
        for i in range(m):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        condition_entropy = sum([len(p) / m * calc_entropy(p) for p in feature_sets.values()])
        return condition_entropy

    # 信息增益
    def info_gain(self, entropy, condition_ent):
        return entropy - condition_ent

    def compute_bestfeature(self, datasets):
        count = len(datasets[0]) - 1
        calc_entropy = self.calc_entropy(datasets)
        best_features = []
        for i in range(count):
            c_info_gain = self.info_gain(calc_entropy, self.condition_entropy(datasets, axis=i))
            best_features.append((i, c_info_gain))
        best_ = max(best_features, key=lambda x: x[1])
        return best_

    def train(self, train_data):
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        features = train_data.columns[:-1]
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return TreeNode(root=True, label=y_train.iloc[0])
        # 2, 若特征集为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return TreeNode(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        max_feature, max_info_gain = self.compute_bestfeature(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return TreeNode(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        # 构建第一个节点
        node_tree = TreeNode(root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)
            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        # pprint.pprint(node_tree.tree)
        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)

datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)
dt = DTree()
tree = dt.fit(data_df)
pprint.pprint(tree)