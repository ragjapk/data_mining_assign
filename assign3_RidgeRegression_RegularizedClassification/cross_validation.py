# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 19:24:46 2018
@author: IIST
"""
import numpy as np
import csv
from gradient_descent import gradient_descent
from gradient_descent import find_rmse_gradient_descent
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import math
def getdata(): 
    reader = csv.reader(open("data2.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    t_rows,t_cols=result.shape
    X=np.delete(result,t_cols-1,1)
    y=np.delete(result, np.s_[0:t_cols-1], axis=1)
    X=np.power(X,8)
    X=np.insert(X,0,1,axis=1)
    #Normalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    mean=np.zeros((X_train.shape[1],1),dtype=float)  
    
    std_dev=np.zeros((X_train.shape[1],1),dtype=float)
    for i in range(1,X_train.shape[1]):
        mean[i]=np.mean(X_train[:,i])
        std_dev[i]=np.std(X_train[:,i])
        
    for i in range(1,X_train.shape[1]):
        X_train[:,i]=(np.subtract(X_train[:,i],mean[i]))/std_dev[i]
    for i in range(1,X_test.shape[1]):
        X_test[:,i]=np.subtract(X_test[:,i],mean[i])/std_dev[i]
    #plot_curve(X_train,y_train)
   
    [alpha,rmse]=k_fold_function(X_train.shape[0],t_cols,X_train,y_train)
    print(alpha)
    w=gradient_descent(X_train,y_train,X_train.shape[0],t_cols,alpha)
    #print(w)
    np.savetxt("w_curve_fitting_8.csv", w, delimiter=",")
    rmse=find_rmse_gradient_descent(X_test,y_test,w)
    print("Final RMSE for 30% data is {}".format(rmse)) 
    

 
def k_fold_function(t_rows,t_cols,X,y):          
    #alpha_list=[0.001,0.00001]   
    min_rmse=np.inf
    #min_alpha=0
    k_folds=10
    k_fold_test_rows=int(t_rows/k_folds)
    k_fold_train_rows=t_rows-k_fold_test_rows
    
    for i in range(-6,1):
        #print(alpha)
        rmse_array=[]
        for k_fold in range (0,k_folds):    
            X_test=X[:k_fold_test_rows]
            y_test=y[:k_fold_test_rows]
            
            X_train=X[k_fold_test_rows:t_rows]
            y_train=y[k_fold_test_rows:t_rows]
            
            w=gradient_descent(X_train,y_train,k_fold_train_rows,t_cols,10**i)
            #print(w)
            rmse_array.append(find_rmse_gradient_descent(X_test,y_test,w))
            X_new=np.concatenate((X_train,X_test))
            y_new=np.concatenate((y_train,y_test))
            X=np.copy(X_new)
            y=np.copy(y_new)
        avg_rmse=np.average(rmse_array)
        if(math.isnan(avg_rmse)):
            avg_rmse=np.inf
        if(avg_rmse<=min_rmse):
            min_rmse=avg_rmse
            min_alpha=10**i
    return (min_alpha,min_rmse)

def plot_curve(X,y):
    reader = csv.reader(open("w_curve_fitting_5.csv", "rt"), delimiter=",")
    x = list(reader)
    w = np.array(x).astype("float")
    fig, ax = plt.subplots()
    y_pred=np.dot(w.T,X.T)
    iterator=np.arange(y_pred.shape[1])
    ax.plot(iterator,y_pred.T)
    ax.scatter(iterator,y,c='r',marker='*',label='actual')
    #ax.legend(loc='best')
    plt.show()
getdata();