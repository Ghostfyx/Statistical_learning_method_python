#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/27 17:52
# @Author  : fanyuexiang
# @Site    : 
# @File    : NaiveBayes.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

iris = load_iris()
df = pd.DataFrame(iris.data)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
data = df.iloc[:100, :]
X = data.iloc[:, :]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


class NaiveBayes:
    def __init__(self):
        self.model = None

    def mean(self, X):
        return np.mean(X)

    def stdev(self, X):
        return np.var(X)

    def gaussian_probability(self, X):
        """
        计算数据分布是高斯分布的概率密度
        :param X:
        """
        std = stdev(X)
        mean = mean(X)
        p = np.exp(-np.power(X - mean, 2) / 2 * std) / np.sqrt(2 * np.pi * std)
        return p

    # 处理X_train
    def summarize(self, values):
        summaries = [(self.mean(i), self.stdev(i)) for i in values]
        return summaries

    def fit(self, X, y):
        labels = set(y.values)
        iris_data = {label: [] for label in labels}
        # 数据的组建
        for x, label in zip(X.values, y):
            iris_data[label].append(x)
        self.model = {label: self.summarize(value) for label, value in iris_data.items()}
        print('model build over')


model = NaiveBayes()
model.fit(X_train, y_train)