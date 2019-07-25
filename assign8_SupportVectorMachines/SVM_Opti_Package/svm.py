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

def create_kernel_matrix(kernel_val,X,Y):
    if(kernel_val=='L'):
        k=X@Y.T
    elif(kernel_val=='P'):
        k=np.power(X@Y.T,2)
    return k

    
def fit(X,y,K,c):
    NUM = X.shape[0]
    print(K.shape)
    P = cvxopt.matrix(K)
    q = cvxopt.matrix(-np.ones((NUM, 1)))
    G = cvxopt.matrix(-np.eye(NUM))
    h = cvxopt.matrix(np.zeros(NUM))
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.zeros(1))
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    alphas[alphas<0]=0
    alphas[alphas>c]=c
    return alphas,sol

def getdata1():
    reader = csv.reader(open("H_new.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    for e in range(-5,1,2):
        c=2**e
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,random_state=42)
    #Creation of inital Kernel Matrix
    K=create_kernel_matrix('L',X_train,X_train)
    Y=np.outer(y_train,y_train)
    K1=K*Y    
    #Get Alphas.
    alphas,sol=fit(X_train,y_train,K1,c)
    y_train=y_train.reshape(y_train.shape[0],1)
    alpha_y= alphas*y_train
    f_train=np.dot(alpha_y.T,K)
    diff=y_train-f_train.T
    cnt = ((alphas >0) & (alphas <c)).shape[0]
    bias=np.sum(diff)/cnt 
    #print(f_train)
    return f_train+bias

def getdata2():
    reader = csv.reader(open("H.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    for e in range(-5,1,2):
        c=2**e
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,random_state=42)
    #Creation of inital Kernel Matrix
    K=create_kernel_matrix('P',X_train,X_train)
    Y=np.outer(y_train,y_train)
    K1=K*Y    
    #Get Alphas.
    alphas,sol=fit(X_train,y_train,K1,c)
    y_train=y_train.reshape(y_train.shape[0],1)
    alpha_y= alphas*y_train
    f_train=np.dot(alpha_y.T,K)
    diff=y_train-f_train.T
    cnt = ((alphas >0) & (alphas <c)).shape[0]
    bias=np.sum(diff)/cnt 
    #print(f_train)
    return f_train+bias

f1=getdata1()
f2=getdata2()
print(np.linalg.norm(f1-f2))