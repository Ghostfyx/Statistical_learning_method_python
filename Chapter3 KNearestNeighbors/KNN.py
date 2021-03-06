#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/25 13:26
# @Author  : fanyuexiang
# @Site    : 
# @File    : KNN.py
# @Software: PyCharm
import math
from collections import Counter

import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def L(x, y, p=2):
    '''
    距离的度量方式：
        p = 1 曼哈顿距离
        p = 2 欧氏距离
        p = inf 闵式距离minkowski_distance
    '''
    if len(x) == len(y) and len(x) >= 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i]-y[i]), p)
    return sum

# 课本例3.1，求解不同距离计算方式（p值），距离x1最近的点？
x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]

# p=1
L12 = L(x1, x2, 1)
L13 = L(x1, x3, 1)
if L12 == L13:
    print('x2 and x3 have the same distance to x1:', L12)
else:
    s = 'x3' if L12 > L13 else 'x2'
    print('%s is closer to x1' %(s))

# p=2
L12 = L(x1, x2, 2)
L13 = L(x1, x3, 2)
if L12 == L13:
    print('x2 and x3 have the same distance to x1:', L12)
else:
    s = 'x3' if L12 > L13 else 'x2'
    print('%s is closer to x1' %(s))

# python实现，遍历所有数据点，找出n个距离最近的点的分类情况，少数服从多数
# data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.groupby()
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
print(df.head(1))
print(df['label'].unique())
# visualization data
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc='upper right')
plt.show()
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class KNN:
    def __init__(self, X_train, y_train, n_neighbors=10, p=2):
        """
        parameter: n_neighbors 临近点个数
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        # 取出n个点
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
        label_map = {}
        for item in knn_list:
            if item[1] in label_map:
                label_map[item[1]] += 1
            else:
                label_map[item[1]] = 1
        sorted(label_map.items(), key=lambda items: items[1])
        return list(label_map.keys())[-1]

    def source(self, X_test, y_test):
        right_count = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        print('the right count is:', (right_count / len(X_test)))

clf = KNN(X_train, y_train)
clf.source(X_test, y_test)