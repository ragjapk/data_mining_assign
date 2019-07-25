1# -*- coding: utf-8 -*-
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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from statistics import mean 
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

def getdata():
    reader = csv.reader(open("data3.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    best_c,bestkernel=kfold_validate(X_train,y_train)
    accuracy=ridge(X_train,y_train,X_test,y_test,best_c,bestkernel)
    print('RMSE obtained is {}'.format(accuracy))
    print('Best kernel is {}'.format(bestkernel))
    print('Best lambda value is {}'.format(best_c))
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
        best_c,bestkernel=kfold_validate(X_train,y_train)
        rmse=ridge(X_train,y_train,X_test,y_test,best_c,bestkernel)
        rmse_arr.append(rmse)
        print(rmse)
    print('Average RMSE obtained is {}'.format(sum(rmse_arr)/len(rmse_arr)))
    print('Best kernel is {}'.format(bestkernel))
    print('Best lambda value is {}'.format(best_c))
    np.savetxt('rmse_ridge.csv',rmse_arr)
    
    
def kfold_validate(X,y):
    skf = KFold(n_splits=5)
    skf.get_n_splits(X, y)
    max_acc=np.inf
    for kernel in ['L','P','E','H',]:
        for e in range(-4,6,2):
            accuracy=[]
            lam=10**e
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]            
                acc=ridge(X_train,y_train,X_test,y_test,lam,kernel)
                accuracy.append(acc)
            accuracy=np.asarray(accuracy)
            avg_acc=np.average(accuracy)
            #print(avg_acc)
            if(avg_acc<max_acc):
                max_acc=avg_acc
                best_c=lam
                bestkernel=kernel
    return best_c,bestkernel 
            
def ridge(X_train,y_train,X_test,y_test,lam,kernel_):   
    y_train=y_train.reshape(y_train.shape[0],1)
    K=create_kernel_matrix(kernel_,X_train,X_train)
    temp1=lam*np.eye(K.shape[0])
    temp2=np.linalg.inv(K+temp1)
    f=np.zeros(X_test.shape[0])
    for i in range(len(X_test)):
        temp3=temp2@K[i]
        f[i]=y_train.T@temp3 
    rms = sqrt(mean_squared_error(y_test, f))
    return rms

getdata2()