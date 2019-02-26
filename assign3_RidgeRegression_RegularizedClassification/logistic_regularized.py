# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:58:34 2018

@author: IIST
"""

import numpy as np
import csv
#from gradient_descent import gradient_descent_ridge_reg_iterative
from gradient_ascent import gradient_ascent_ridge_reg
from sklearn.model_selection import train_test_split
import math
from performance_calculator import find_performance_measures
from sklearn.model_selection import StratifiedKFold
from performance_calculator import get_final_performance
import time
def getdata(): 
    reader = csv.reader(open("data1.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    rows,cols=result.shape
    X=np.delete(result,cols-1,1)
    y=np.delete(result, np.s_[0:cols-1], axis=1)
    X=np.insert(X,0,1,axis=1)                
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)

    mean=np.zeros((X_train.shape[1],1),dtype=float)  
    
    std_dev=np.zeros((X_train.shape[1],1),dtype=float)
    for i in range(1,X_train.shape[1]):
        mean[i]=np.mean(X_train[:,i])
        std_dev[i]=np.std(X_train[:,i])
        
    for i in range(1,X_train.shape[1]):
        X_train[:,i]=(np.subtract(X_train[:,i],mean[i]))/std_dev[i]
        
    alpha,lambdaa,acc=k_fold_function(X_train.shape[0],X_train.shape[1],X_train,y_train)
    start=time.time()
    w=gradient_ascent_ridge_reg(X_train,y_train,X_train.shape[0],X_train.shape[1],alpha,lambdaa)
    end=time.time()
    print(end-start)
    #Normalize X_test:
    for i in range(1,X_test.shape[1]):
        X_test[:,i]=np.subtract(X_test[:,i],mean[i])/std_dev[i]
    np.savetxt("w_logistic_reg_adultdata.csv", w, delimiter=",")       
    tp,fp,fn,tn=find_performance_measures(X_test,y_test,w)
    sensitivity,specificity,precision,accuracy,f_measure=get_final_performance(tp,fp,tn,fn)
    print('These are the values:{},{},{}'.format(acc,alpha,lambdaa))
    print("Accuracy on test data is {}".format(accuracy))
    print("Sensitivity on test data is {}".format(sensitivity))
    print("Specificity on test data is {}".format(specificity))
    print("Precision on test data is {}".format(precision))
    print("F_Measure on test data is {}".format(f_measure))
    
def k_fold_function(t_rows,t_cols,X,y):          
    best_alpha=0
    best_alpha_final=0
    best_lambda=0
    avg_accuracy=np.NINF
    max_accuracy=np.NINF
    max_accuracy_final=np.NINF
    skf = StratifiedKFold(n_splits=3)
    skf.get_n_splits(X, y)
    for l in range(-9,-1):
        for i in range(-5,-1): 
            accuracy_array=[]                   
            for train_index, test_index in skf.split(X, y): 
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]                 
                w=gradient_ascent_ridge_reg(X_train,y_train,X_train.shape[0],t_cols,10**i,2**i)
                tp,fp,fn,tn=find_performance_measures(X_test,y_test,w)
                sensitivity,specificity,precision,accuracy,f_measure=get_final_performance(tp,fp,tn,fn)
                accuracy_array.append(f_measure)
                X_new=np.concatenate((X_train,X_test))
                y_new=np.concatenate((y_train,y_test))
                X=np.copy(X_new)
                y=np.copy(y_new)
            avg_accuracy=np.average(accuracy_array)
            if(math.isnan(avg_accuracy)):
                avg_accuracy=np.inf
            if(avg_accuracy>max_accuracy):
                max_accuracy=avg_accuracy
                best_alpha=10**i
        if(max_accuracy>max_accuracy_final):
            max_accuracy_final=max_accuracy
            best_lambda=2**l
            best_alpha_final=best_alpha
    return (best_alpha_final,best_lambda,max_accuracy_final)

getdata();