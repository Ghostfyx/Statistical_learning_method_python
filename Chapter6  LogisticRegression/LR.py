#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/8 15:05
# @Author  : fanyuexiang
# @Site    : 
# @File    : LR.py
# @Software: PyCharm
from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import optimize as op
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data)
    df['target'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    return df
df = create_data()
df.head()
X = df.iloc[:100, [0, 1]]
y = df.iloc[:100, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)


class LogisticReressionClassifier:
    def __init__(self, max_iter=2000, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        g = 1 / (1 + np.exp(-z))
        return g

    def cost_function(self, theta, x, y):
        m = y.size
        z = x.dot(theta)
        g = self.sigmoid(z)
        J = -1 / m * (y.T.dot(np.log(g)) + (1 - y).T.dot(np.log((1 - g))))
        return float(J)

    def gradientFunction(self, theta, x, y):
        m = len(y)  # number of training examples
        grad = 1 / m * (x.T.dot(self.sigmoid(x.dot(theta)) - y))
        return grad

    def fit(self, x, y):
        m, n = x.shape
        x = np.concatenate((np.ones((m, 1)), x), axis=1)
        self.theta = np.zeros(n + 1)
        min_cost_result = op.minimize(self.cost_function, self.theta, (x, y), 'BFGS', jac=self.gradientFunction)
        self.theta = min_cost_result.x
        print('theta is', self.theta)

    def score(self, x_test, y_test):
        right = 0
        m, n = x_test.shape
        a = np.ones((m, 1))
        x_test = np.concatenate((a, x_test), axis=1)
        for x, y in zip(x_test, y_test):
            result = self.theta.dot(x)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        accuracy = right / len(y_test)
        return accuracy

lr_clf = LogisticReressionClassifier()
lr_clf.fit(X_train, y_train)
accuracy = lr_clf.score(X_test, y_test)
print('the LR Classifier accuracy rate is %.2f', (accuracy))