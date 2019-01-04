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
import jupyter_contrib_nbextensions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

iris = load_iris()
df = pd.DataFrame(iris.data)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
data = df.iloc[:100, :]
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test

class NaiveBayes:
    def __init__(self):
        self.model = None
        self.priorProbability = None

    def mean(self, X):
        return np.mean(X)

    def stdev(self, X):
        return np.var(X)

    def gaussian_probability(self, X, mean, std):
        """
        计算数据分布是高斯分布的概率密度
        :param X:
        """
        p = np.exp(-np.power(X - mean, 2) / 2 * std) / np.sqrt(2 * np.pi * std)
        return p

    def prior_probability(self, n):
        """
        计算所有类别的先验概率 P(Y=C_k)
        :param n:
        :return:
        """
        priorProbability = {}
        for label, values in self.model.items():
            priorProbability[label] = len(values) / n
        return priorProbability

    # 处理X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    def calculate_probabilities(self, X):
        probabilities = {}
        for key, values in self.model.items():
            probabilities[key] = 1
            # 计算先验条件概率分布
            for i in range(len(values)):
                mean, stdev = values[i]
                probabilitie = self.priorProbability[key] * self.gaussian_probability(X[i], mean, stdev)
                probabilities[key] *= probabilitie
        return probabilities

    def fit(self, X, y):
        n = X.shape[0]
        labels = set(y.values)
        iris_data = {label: [] for label in labels}
        # 数据的组建
        for x, label in zip(X.values, y):
            iris_data[label].append(x)
        self.model = {label: self.summarize(value) for label, value in iris_data.items()}
        self.priorProbability = self.prior_probability(n)
        print('model build over')

    def predict(self, X_test):
        probabilities = self.calculate_probabilities(X_test)
        sorted(probabilities.items(), key=lambda item: item[1])
        return list(probabilities.keys())[-1]

    def score(self, X_test, y_test):
        right_count = 0
        for x_item, y_item in zip(X_test.values, y_test.values):
            label = self.predict(x_item)
            if label == y_item:
                right_count += 1
        print(right_count)
        print(len(X_test))
        print(right_count / len(X_test))

model = NaiveBayes()
model.fit(X_train, y_train)
model.score(X_test, y_test)
