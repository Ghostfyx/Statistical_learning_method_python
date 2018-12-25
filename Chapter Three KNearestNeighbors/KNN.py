#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/25 13:26
# @Author  : fanyuexiang
# @Site    : 
# @File    : KNN.py
# @Software: PyCharm
import math
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


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
print(X)