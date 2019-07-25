# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 01:00:06 2019

@author: IIST
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

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

def fit2(X,y,K,c,epsilon,lr):
    betas=np.zeros(X.shape[0])
    beta_new=np.zeros(X.shape[0])
    err=1
    itera=0
    while err>1e-2 and itera<200:
        itera=itera+1
        for i in range(X.shape[0]):        
            temp=betas@K[i]
            grad=y[i]-epsilon*np.sign(betas[i])-temp
            beta_new[i]=betas[i]-lr*grad
            err=np.linalg.norm(beta_new[i]-betas[i])
            betas=np.copy(beta_new)
            print(err)                
    betas[betas<-c]=-c
    betas[betas>c]=c
    return betas

def getdata():
    reader = csv.reader(open("data3.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    best_c,besteps,best_lr,best_k=kfold_validate(X_train,y_train)
    accuracy=adatron(X_train,y_train,X_test,y_test,best_c,besteps,best_k,best_lr)
    print('RMSE obtained is {}'.format(accuracy))
    print('C obtained after cross validation is {}'.format(best_c))
    print('Best epsilon obtained after cross validation is {}'.format(besteps))
    print('Best kernel is {}'.format(best_k))
    #plot_data_with_labels(X_train,y_train,X_test,y_test,best_c,alphas,bias,bestkernel,sv)

def getdata2():
    reader = csv.reader(open("data2.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    rmse_arr=[]
    holdout_size=5
    for i in range(0,holdout_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
        best_c,besteps,best_lr,best_k=kfold_validate(X_train,y_train)
        rmse=adatron(X_train,y_train,X_test,y_test,best_c,besteps,best_k,best_lr)
        rmse_arr.append(rmse)
        print(rmse)
    print('Average RMSE obtained is {}'.format(sum(rmse_arr)/len(rmse_arr)))
    print('C obtained after cross validation is {}'.format(best_c))
    print('Best epsilon obtained after cross validation is {}'.format(besteps))
    print('Best kernel is {}'.format(best_k))
    np.savetxt('rmse_iterative.csv',rmse_arr)
    
    #
def kfold_validate(X,y):
    skf = KFold(n_splits=5)
    skf.get_n_splits(X, y)
    max_acc=np.inf
    kernel='E'
    for eps in range(-6,-2,1):
        for e in range(3,5,2):
            for r in range(-4,-2,2):
                for kernel in ['E','H','P','L']:
                    accuracy=[]
                    c=10**e
                    lr=10**r
                    epsilon=10**eps
                    for train_index, test_index in skf.split(X, y):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]            
                        acc=adatron(X_train,y_train,X_test,y_test,c,epsilon,kernel,lr)
                        accuracy.append(acc)
                    accuracy=np.asarray(accuracy)
                    avg_acc=np.average(accuracy)
                    print(avg_acc)
                    if(avg_acc<max_acc):
                        max_acc=avg_acc
                        best_lr=lr
                        besteps=epsilon
                        best_c=c
                        best_kernel=kernel
    return best_c,besteps,best_lr,best_kernel  
            
def adatron(X_train,y_train,X_test,y_test,c,epsilon,kernel_,lr):    
    #Creation of inital Kernel Matrix
    y_train=y_train.reshape(y_train.shape[0],1)
    K=create_kernel_matrix(kernel_,X_train,X_train) 
    #Get Alphas.
    betas=fit2(X_train,y_train,K,c,epsilon,lr)
    betas=betas.reshape(betas.shape[0],1)
    #free_sv = np.logical_and(alphas > 0, alphas < c).reshape(-1)
    #free_sv_alpha = alphas[free_sv]
    #s_v = X_train[free_sv]    
    f_train=np.dot(betas.T,K)
    diff=y_train-f_train.T
    cnt = ((betas >-c) & (betas <c)).shape[0]
    bias=np.sum(diff)/cnt    
    f=np.zeros((X_test.shape[0]))   
    
    #Creation of Kernel Matrix for Testing
    K1=create_kernel_matrix(kernel_,X_train,X_test)    
    f=np.dot(betas.T,K1)
    f=f.reshape(f.shape[1],1)
    f_tilda=f+bias
    rms = sqrt(mean_squared_error(y_test, f_tilda))
    print(rms)
    return rms


getdata2()