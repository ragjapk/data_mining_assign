# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:03:06 2019

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
        sigma =100
        k= np.exp((-1/(2*sigma**2))*np.linalg.norm(X[:,None]-Y, axis=2))
    elif(kernel_val=='H'):
        c=10
        k = 1/np.sqrt(np.linalg.norm(X[:,None]-Y, axis=2)+c)
    return k
    
def fit2(X,y,K,c):
    alphas=np.random.randn(X.shape[0])
    etas2=np.zeros(X.shape[0])
    alpha_new=np.zeros(X.shape[0])
    etas=np.diagonal(K)
    etas=1/etas
    err=1
    itere=0
    while itere<20:
        itere=itere+1
        for i in range(X.shape[0]):  
            alph_y=np.multiply(alphas,y)
            temp=np.dot(alph_y,K[i])
            #print(temp)
            alpha_new[i]=alphas[i]+(etas[i]*(1-(y[i]*temp)))
        print(err)
        err=np.linalg.norm(alpha_new-alphas)
        alphas=np.copy(alpha_new)
    alphas[alphas<0]=0
    alphas[alphas>c]=c
    return alphas

def getdata1():
    reader = csv.reader(open("data6.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    for e in range(-5,1,2):
        c=2**e
    c=10**3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,random_state=42)
    #Creation of inital Kernel Matrix
    K=create_kernel_matrix('L',X_train,X_train)    
    #Get Alphas.
    alphas=fit2(X_train,y_train,K,c)
    y_train=y_train.reshape(y_train.shape[0],1)
    alpha_y= alphas*y_train
    f_train=np.dot(alpha_y.T,K)
    diff=y_train-f_train.T
    cnt = ((alphas >0) & (alphas <c)).shape[0]
    bias=np.sum(diff)/cnt 
    #print(f_train)
    return f_train+bias,y_test
    

f1,y_test=getdata1()
#prediction=np.sign(f1)  
#score=accuracy_score(y_test, prediction, sample_weight=None)
#print(score)