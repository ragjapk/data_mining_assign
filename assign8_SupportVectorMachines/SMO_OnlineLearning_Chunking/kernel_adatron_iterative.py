# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:18:42 2019

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
import random
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

    
def plot_data_with_labels(X_train,y_train,X_test,y_test,c,alphas,bias,kernel,sv):
    x_min=X_test[:, 0].min() - 1
    x_max =X_test[:, 0].max() + 1
    y_min = X_test[:, 1].min() - 1
    y_max = X_test[:, 1].max() + 1
    h = 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    X_p = np.c_[xx.ravel(), yy.ravel()]
    pred=test_svm(X_train,y_train,X_p,c,alphas,bias,kernel)
    y_p= pred.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx,yy,y_p,cmap='Accent',alpha=0.7)
    X_test_p = X_test[y_test == 1]
    X_test_n = X_test[y_test == -1]
    plt.scatter(X_test_p[:,0],X_test_p[:,1],c='blue')
    plt.scatter(X_test_n[:,0],X_test_n[:,1],c='red')
    plt.scatter(sv[:,0],sv[:,1],c='green')
    plt.show()

def fit2(X,y,K,c):
    alphas=np.zeros(X.shape[0])
    alpha_new=np.zeros(X.shape[0])
    etas=np.diagonal(K)
    etas=1/etas
    err=1
    while err>1e-2:
        for i in range(X.shape[0]):            
            temp=(alphas*y)@K[i]
            alpha_new[i]=alphas[i]+(etas[i]*(1-(y[i]*temp)))
            err=np.linalg.norm(alpha_new[i]-alphas[i])
            alphas=np.copy(alpha_new)
        print(err)                
    alphas[alphas<0]=0
    alphas[alphas>c]=c
    return alphas

def getdata():
    reader = csv.reader(open("data6.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)
    best_c,bestkernel=kfold_validate(X_train,y_train)
    accuracy,f,bias,alphas,pred=adatron(X_train,y_train,X_test,y_test,best_c,bestkernel)
    print('Value of bias is {}'.format(bias))
    print('Accuracy obtained is {}'.format(accuracy))
    print('C obtained after cross validation is {}'.format(best_c))
    print('Best kernel is {}'.format(bestkernel))
    #plot_data_with_labels(X_train,y_train,X_test,y_test,best_c,alphas,bias,bestkernel,sv)
    
    #
def kfold_validate(X,y):
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X, y)
    max_acc=np.NINF
    for kernel in ['L','P','E','H',]:
        for e in range(3,6,2):
            accuracy=[]
            c=10**e
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]            
                acc,f,bias,alphas,pred=adatron(X_train,y_train,X_test,y_test,c,kernel)
                accuracy.append(acc)
            accuracy=np.asarray(accuracy)
            avg_acc=np.average(accuracy)
            print(avg_acc)
            if(avg_acc>max_acc):
                max_acc=avg_acc
                best_c=c
                bestkernel=kernel
    return best_c,bestkernel  
            
def adatron(X_train,y_train,X_test,y_test,c,kernel_):    
    #Creation of inital Kernel Matrix
    K=create_kernel_matrix(kernel_,X_train,X_train) 
    #Get Alphas.
    alphas=fit2(X_train,y_train,K,c)
    alphas=alphas.reshape(alphas.shape[0],1)
    #free_sv = np.logical_and(alphas > 0, alphas < c).reshape(-1)
    #free_sv_alpha = alphas[free_sv]
    #s_v = X_train[free_sv]    
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
    return score,f,bias,alphas,prediction


getdata()