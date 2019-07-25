# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 23:30:09 2019

@author: IIST
"""
import numpy as np
import csv
import cvxopt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
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

def fit2(X_train,y_train,kernel_,c,M,S):
    k=6
    alphass=np.zeros((X_train.shape[0],1))
    while(True):        
        K=create_kernel_matrix(kernel_,X_train[M],X_train[M])
        Y=np.outer(y_train[M],y_train[M])
        K1=K*Y 
        alphas,sol=fit(X_train[M],y_train[M],K1,c)
        alphas=alphas.reshape(alphas.shape[0],)
        sv = M[alphas>0]
        alphas=alphas.reshape(alphas.shape[0],1)
        #sv = np.where( np.logical_and( alphas > 0, alphas < c) )[0] 
       
        y_M=y_train[M]
        y_M=y_M.reshape(y_M.shape[0],1)
        alpha_y= alphas*y_M
        f_train=np.dot(alpha_y.T,K)
        diff=y_M-f_train.T
        cnt = ((alphas >0) & (alphas <c)).shape[0]
        bias=np.sum(diff)/cnt  
        S_M = np.setxor1d(S, M)
        print(len(M))
        print(len(S_M))
        #print(S_M.shape[0])
        #a=X_train[free_sv]
        #b=X_train[S_M]
        K1=create_kernel_matrix(kernel_,X_train[M],X_train[S_M])
        
        f=np.dot(alpha_y.T,K1)
        f=f.reshape(f.shape[1],1)
        f_tilda=f+bias
        prediction=np.sign(f_tilda) 
        prediction=prediction.reshape(prediction.shape[0],1)
        y_SM=y_train[S_M]
        y_SM=y_SM.reshape(y_SM.shape[0],1)
        p=np.multiply(y_SM,prediction)
        ind=np.where(p==-1)[0]
        error=S_M[ind]
        #print(len(error))
        if (len(error) == 0):
            break
        elif len(error) > k :
            error=error[:k]
        #print(error)
        M=np.array(list(set(sv).union(error)))
        #print(M)
    alphas=alphas.reshape(alphas.shape[0],1)
    alphass[M]=alphas
    return alphass
    
def fit(X,y,K,c):
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
    alphas[alphas<0]=0
    alphas[alphas>c]=c
    return alphas,sol

def getdata():
    reader = csv.reader(open("data6.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)
    best_c,bestkernel=kfold_validate(X_train,y_train)
    accuracy=svm(X_train,y_train,X_test,y_test,best_c,bestkernel)
    print('Accuracy obtained is {}'.format(accuracy))
    print('C obtained after cross validation is {}'.format(best_c))
    print('Best kernel is {}'.format(bestkernel))

def kfold_validate(X,y):
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X, y)
    max_acc=np.NINF
    for kernel in ['L','P','E','H',]:
        for e in range(-4,-1,2):
            accuracy=[]
            c=10**e
            for train_index, test_index in skf.split(X, y):
                print(c,kernel)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]            
                acc=svm(X_train,y_train,X_test,y_test,c,kernel)
                accuracy.append(acc)
            accuracy=np.asarray(accuracy)
            avg_acc=np.average(accuracy)
            #print(avg_acc)
            if(avg_acc>max_acc):
                max_acc=avg_acc
                best_c=c
                bestkernel=kernel
    return best_c,bestkernel  
            
def svm(X_train,y_train,X_test,y_test,c,kernel_):  
    K=create_kernel_matrix(kernel_,X_train,X_train)
        
    indexes = np.arange(0, X_train.shape[0],1)
    
    #subsize=int(0.4*indexes.shape[0])
    subsize=20
    M=np.random.choice(indexes,subsize)
    alphas=fit2(X_train,y_train,kernel_,c,M,indexes) 
    
    y_train=y_train.reshape(y_train.shape[0],1)
    alpha_y= alphas*y_train
    f_train=np.dot(alpha_y.T,K)
    diff=y_train-f_train.T
    cnt = ((alphas >0) & (alphas <c)).shape[0]
    bias=np.sum(diff)/cnt    
    f=np.zeros((X_test.shape[0]))   
    
    #Creation of Kernel Matrix for Testing
    K1=create_kernel_matrix(kernel_,X_train,X_test)
    
    f=np.dot(alpha_y.T,K1)
    f=f.reshape(f.shape[1],1)
    f_tilda=f+bias
    prediction=np.sign(f_tilda)  
    score=accuracy_score(y_test, prediction, sample_weight=None)
    return score
getdata()