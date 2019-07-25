# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:50:05 2019

@author: IIST
"""
import numpy as np
import csv
import cvxopt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def create_kernel_matrix(kernel_val,X):
    if(kernel_val=='L'):
        k=X@X.T
    elif(kernel_val=='P'):
        k=np.power(X@X.T,2)
    return k

COLORS=['red','blue']

def plot_separator(ax, w, b): 
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    x = np.arange(0, 10)
    ax.plot(x, x * slope + intercept, 'k-')
    
def plot_data_with_labels(x, y, ax):
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        ax.scatter(x_sub[:, 0], x_sub[:, 1], c = COLORS[li])
    
def fit(X,y,K):
    NUM = X.shape[0]
    P = cvxopt.matrix(K)
    q = cvxopt.matrix(-np.ones((NUM, 1)))
    G = cvxopt.matrix(-np.eye(NUM))
    h = cvxopt.matrix(np.zeros(NUM))
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.zeros(1))
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

reader = csv.reader(open("data1.csv", "rt"), delimiter=",")
x = list(reader)
H = np.array(x).astype("float")
X=np.delete(H,-1,axis=1)
y=np.take(H, -1, axis=1)
kernel_='L'
K=create_kernel_matrix(kernel_,X)
Y=np.outer(y,y)
K=K*Y
alphas=fit(X,y,K)
y=y.reshape(1,y.shape[0])
alphas=alphas.reshape(1,alphas.shape[0])
alph_y=alphas*y
f=np.dot(alph_y,K)

cond = (alphas > 1e-4).reshape(-1)
y=y.reshape(y.shape[1],1)
f=f.reshape(f.shape[1],1)
b = y[cond] - f[cond]
bias=np.average(b)
f_tilda=f+bias
prediction=np.sign(f_tilda)
    

score=accuracy_score(y, prediction, normalize=True, sample_weight=None)
print(score)    

print(classification_report(y, prediction))
'''temp=alphas*y
np.dot(temp,K)
f = np.sum(alphas * K , axis = 0)
# get bias
cond = (alphas > 1e-4).reshape(-1)
b = y[cond] - np.dot(X[cond], f)
bias = b[0]

F=f+bias
prediction=np.sign(F)
print(prediction)

    # normalize
norm = np.linalg.norm(w)
w, bias = w / norm, bias / norm

    # show data and w
fig, ax = plt.subplots()
#plot_separator(ax, w, bias)
plot_data_with_labels(X, y, ax)
plt.show()
'''