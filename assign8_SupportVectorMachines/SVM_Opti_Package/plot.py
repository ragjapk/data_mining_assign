# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 22:43:26 2019

@author: IIST
"""

import numpy as np
import pickle, sys
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import csv
COLORS = ['red', 'blue']

def read_data(f):
    with open(f, 'rb') as f:
        data = pickle.load(f)
    x, y = data[0], data[1]
    return x, y

def fit(x, y):
    NUM = x.shape[0]
    # we'll solve the dual
    # obtain the kernel
    K = y[:, None] * x
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((NUM, 1)))
    G = matrix(-np.eye(NUM))
    h = matrix(np.zeros(NUM))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


def plot(X,y, w, b):
    X_test_p = X[y == 1]
    X_test_n = X[y == -1]
    bias=np.average(b)
    print(bias)
    print(w)
    x_min=X[:, 0].min()- 1
    x_max =X[:, 0].max() + 1
    h=0.5
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    
    z = (-w[0] * xx - w[1] * yy + bias) * 1. /w[2]
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, z)
    for i in range(1,len(X)):
        if(y[i]==-1):
            ax.scatter(X[i,0],X[i,1],X[i,2],c='black')
        if(y[i]==1):
            ax.scatter(X[i,0],X[i,1],X[i,2],c='red')
    
if __name__ == '__main__':
    reader = csv.reader(open("Data4_new.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    # fit svm classifier
    alphas = fit(X, y)

    # get weights
    w = np.sum(alphas * y[:, None] * X, axis = 0)
    # get bias
    cond = (alphas > 0).reshape(-1)
    b = y[cond] - np.dot(X[cond], w)
    bias = b[0]

    # normalize
    norm = np.linalg.norm(w)
    w, bias = w / norm, bias / norm

    plot(X, y, w, b)
    plt.show()