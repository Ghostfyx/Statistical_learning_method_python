#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/19 13:09
# @Author  : fanyuexiang
# @Site    : 使用最小二乘法拟和曲线 P11 例题1.1 过拟合和模型的选择；
# 我们用目标函数y=sin2{π}x, 加上一个正太分布的噪音干扰，用多项式去拟合;现在假定数据集个数为10，多项式次数未0-9
# @File    : least_sqaure_method.py
# @Software: PyCharm
import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret
x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
# 加上正态分布噪音的目标函数的值
y_ = real_func(x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]


def fitting(M=0):
    """
    M    为 多项式的次数
    """
    # 随机初始化多项式参数
    print('m:', M)
    p_init = np.random.rand(M + 1)
    print('p_init:', p_init)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    print('Fitting Parameters:', p_lsq[0])

    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq

p_lsq_0 = fitting(M=0)
p_lsq_0 = fitting(M=1)
p_lsq_0 = fitting(M=2)
p_lsq_0 = fitting(M=3)
p_lsq_0 = fitting(M=6)
p_lsq_0 = fitting(M=9)
# 引入正则化项
regularization = 0.0001
def residuals_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))  # L2范数作为正则化项