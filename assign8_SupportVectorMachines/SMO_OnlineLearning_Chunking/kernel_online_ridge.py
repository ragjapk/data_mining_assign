# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 02:25:56 2019

@author: IIST
"""

import numpy as np
import csv
import cvxopt
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
        sigma =40
        k= np.exp((-1/(2*sigma**2))*np.linalg.norm(X[:,None]-Y, axis=2))
    elif(kernel_val=='H'):
        c=10
        k = 1/np.sqrt(np.linalg.norm(X[:,None]-Y, axis=2)+c)
    return k

def fit2(X,y,K,eta,lamb):
    alphas=np.zeros(X.shape[0])
    alphas_new=np.zeros(X.shape[0])
    for i in range(X.shape[0]):            
        alphas_new[0:i]=alphas[0:i]*(1-eta*lamb)
        f=np.dot(alphas_new.T,K[i])
        diff=f-y[i]
        alphas_new[i]=-eta*diff
        alphas=np.copy(alphas_new)               
    return alphas

def getdata():
    reader = csv.reader(open("data3.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    bestkernel,eta,lamb=kfold_validate(X_train,y_train)
    accuracy=adatron(X_train,y_train,X_test,y_test,bestkernel,eta,lamb)
    print('RMS obtained is {}'.format(accuracy))
    print('Learning Rate obtained after cross validation is {}'.format(eta))
    print('Best kernel is {}'.format(bestkernel))
    print('Best lambda value is {}'.format(lamb))
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
        bestkernel,eta,lamb=kfold_validate(X_train,y_train)
        rmse=adatron(X_train,y_train,X_test,y_test,bestkernel,eta,lamb)
        rmse_arr.append(rmse)
        print(rmse)
    print('Average RMSE obtained is {}'.format(sum(rmse_arr)/len(rmse_arr)))
    print('Learning Rate obtained after cross validation is {}'.format(eta))
    print('Best kernel is {}'.format(bestkernel))
    print('Best lambda value is {}'.format(lamb))
    np.savetxt('rmse_online.csv',rmse_arr)
    
    #
def kfold_validate(X,y):
    skf = KFold(n_splits=5)
    skf.get_n_splits(X, y)
    max_acc=np.inf
    for kernel in ['L','P','E','H',]:
        for et in (-5,0,1):
            for lam in range(-4,0,1):
                accuracy=[]
                eta=10**et
                lamb=10**lam
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]            
                    acc=adatron(X_train,y_train,X_test,y_test,kernel,eta,lamb)
                    accuracy.append(acc)
                accuracy=np.asarray(accuracy)
                try:
                    avg_acc=np.average(accuracy)
                except:
                    accuracy = accuracy[np.logical_not(np.isnan(accuracy))]
                    accuracy = accuracy[np.logical_not(np.isinf(accuracy))]
                    avg_acc=np.average(accuracy)
                if(avg_acc<max_acc):
                    max_acc=avg_acc
                    bestkernel=kernel
                    best_eta=eta
                    best_lamb=lamb
    return bestkernel,best_eta,best_lamb
            
def adatron(X_train,y_train,X_test,y_test,kernel_,eta,lamb):    
       #Creation of inital Kernel Matrix
    y_train=y_train.reshape(y_train.shape[0],1)
    K=create_kernel_matrix(kernel_,X_train,X_train) 
    #Get Alphas.
    alphas=fit2(X_train,y_train,K,eta,lamb)
    alphas=alphas.reshape(alphas.shape[0],1)
    #free_sv = np.logical_and(alphas > 0, alphas < c).reshape(-1)
    #free_sv_alpha = alphas[free_sv]
    #s_v = X_train[free_sv]     
    
    #Creation of Kernel Matrix for Testing
    K1=create_kernel_matrix(kernel_,X_train,X_test)    
    f=np.dot(alphas.T,K1)
    f=f.reshape(f.shape[1],1)
    f_tilda=f
    rms = sqrt(mean_squared_error(y_test, f_tilda))
    return rms


getdata2()