#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 15:01
# @Author  : fanyuexiang
# @Site    : 感知机
# @File    : perceptron.py
# @Software: PyCharm
'''
假设训练数据集是线性可分的，感知机学习的目标就是求得一个能够将训练数据集中正负实例完全分开的分类超平面，
为了找到分类超平面，即确定感知机模型中的参数w和b，需要定义一个损失函数并通过将损失函数最小化来求w和b。
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
print('begin load data')
iris = load_iris()
df = pd.DataFrame(iris.data)
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
df['labels'] = iris.target
print('begin analyze data')
print(df['labels'].value_counts())
print('begin visualization data')
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label ='setosa')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label ='versicolor')
plt.scatter(df[100:150]['sepal length'], df[100:150]['sepal width'], label ='virginica')
plt.legend()
plt.show()
print('begin train dataSet')
X = df.iloc[:100, [0,1]]
y = df.iloc[:100, -1]
y = np.array([1 if i == 1 else -1 for i in y])
print('build Perceptron Model')
# 数据线性可分，二分类数据
class Model_Perceptron:
    def __init__(self,w,b):
        self.w = w
        self.b = b

    def sign(self, X, w, b):
        y = X.dot(w)+b
        return y

    def fit(self, X, y, alpha=0.1):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i in range(X.shape[0]):
                x_item = X.iloc[i]
                y_item = y[i]
                if y_item*self.sign(x_item, self.w, self.b) <= 0:
                    self.w = self.w + alpha*np.dot(x_item, y_item)
                    self.b = self.b + alpha*y_item
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong=True
                break
                print('model build done')


print('begin train data')
m,n = X.shape
w = np.zeros(n)
b = 0
perceptron = Model_Perceptron(w=w, b=b)
perceptron.fit(X, y)
print('parameters: w_0=%.2f,w_1%.2f'%(perceptron.w[0],perceptron.w[1]))
print('begin plot classify decision boundary')
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label ='setosa')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label ='versicolor')
x_points = np.linspace(np.min(X['sepal length']), np.max(X['sepal length']),10)
y_ = -(perceptron.w[0]*x_points + perceptron.b)/perceptron.w[1]
plt.plot(x_points, y_, label='model')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
print('begin use sklearn Perceptron')
sklearn_perceptron = Perceptron(fit_intercept=False, alpha=0.1, max_iter=1000, shuffle=False)
sklearn_perceptron.fit(X, y)
print('use sklearn Perceptorn parameters w:', sklearn_perceptron.coef_)
print('use sklearn Perceptorn parameters b:', sklearn_perceptron.intercept_)
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label ='setosa')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label ='versicolor')
x_points = np.linspace(np.min(X['sepal length']), np.max(X['sepal length']),10)
y_ = -(sklearn_perceptron.coef_[0][0]*x_points + sklearn_perceptron.intercept_)/sklearn_perceptron.coef_[0][1]
plt.plot(x_points, y_, label='model')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
