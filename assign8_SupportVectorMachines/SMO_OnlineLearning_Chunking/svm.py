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
    elif(kernel_val=='E'):
        sigma =40
        k= np.exp((-1/(2*sigma**2))*np.linalg.norm(X[:,None]-Y, axis=2))
    elif(kernel_val=='H'):
        c=10
        k = 1/np.sqrt(np.linalg.norm(X[:,None]-Y, axis=2)+c)
    return k
    
def fit2(X,y,K,c):
    alphas=np.zeros(X.shape[0])
    alpha_new=np.zeros(X.shape[0])
    etas=np.diagonal(K)
    etas=1/etas
    err=1
    while err>1e-3:
        for i in range(X.shape[0]):            
            temp=(alphas*y)@K[i]
            alpha_new[i]=alphas[i]+(etas[i]*(1-(y[i]*temp)))
            err=np.linalg.norm(alpha_new[i]-alphas[i])
            alphas=np.copy(alpha_new)
    print(err)                
    alphas[alphas<0]=0
    alphas[alphas>c]=c
    return alphas

def getdata1():
    reader = csv.reader(open("data5.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    c=10**3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)
    K=create_kernel_matrix('H',X_train,X_train)    
    alphas=fit2(X_train,y_train,K,c)
    
    y_train=y_train.reshape(y_train.shape[0],1)
    alphas=alphas.reshape(alphas.shape[0],1)
    alpha_y= alphas*y_train
    f_train=np.dot(alpha_y.T,K)
    diff=y_train-f_train.T
    cnt = ((alphas >0) & (alphas <c)).shape[0]
    bias=np.sum(diff)/cnt 
    return bias,y_test,X_test,alphas,y_train,X_train
    

bias,y_test,X_test,alphas,y_train,X_train=getdata1()
print(bias)
alphas=alphas.reshape(alphas.shape[0],1)
y_train=y_train.reshape(y_train.shape[0],1)
K1=create_kernel_matrix('H',X_train,X_test)
alpha_y= alphas*y_train
f=np.dot(alpha_y.T,K1)
f=f.reshape(f.shape[1],1)
f_tilda=f+bias
print(f_tilda)
prediction=np.sign(f_tilda)  
score=accuracy_score(y_test, prediction, sample_weight=None) 
#score=accuracy_score(y_test, prediction, sample_weight=None)
print(score)