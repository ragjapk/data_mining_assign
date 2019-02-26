# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:05:09 2018

@author: IIST
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 19:24:46 2018
@author: IIST
"""
import numpy as np
import csv
from gradient_descent import find_rmse_gradient_descent
from gradient_descent import gradient_descent_ridge_reg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
def getdata(): 
    reader = csv.reader(open("data1.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    t_rows,t_cols=result.shape
    preprocessor(result,t_cols,t_rows)
    
def preprocessor(result,t_cols,t_rows):
    X=np.delete(result,t_cols-1,1)
    y=np.delete(result, np.s_[0:t_cols-1], axis=1)
    #X=np.power(X,8)
    X=np.insert(X,0,1,axis=1)
    #Normalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    #mean=np.zeros((X_train.shape[1],1),dtype=float)  
    
    #std_dev=np.zeros((X_train.shape[1],1),dtype=float)
    #for i in range(1,X_train.shape[1]):
    #    mean[i]=np.mean(X_train[:,i])
    #    std_dev[i]=np.std(X_train[:,i])
        
    #for i in range(1,X_train.shape[1]):
    #    X_train[:,i]=(np.subtract(X_train[:,i],mean[i]))/std_dev[i]
    #for i in range(1,X_test.shape[1]):
    #    X_test[:,i]=np.subtract(X_test[:,i],mean[i])/std_dev[i]
    holdout(X_train,X_test,y_train,y_test,t_cols)
    
def holdout(X_train,X_test,y_train,y_test,t_cols):
    k_fold_function(X_train.shape[0],t_cols,X_train,y_train)   
    
def k_fold_function(t_rows,t_cols,X,y):          
    #min_alpha=0
    rms_train=[]
    rms_val=[]
    k_folds=5
    k_fold_test_rows=int(t_rows/k_folds)
    k_fold_train_rows=t_rows-k_fold_test_rows
    for l in range(-16,0):
        min_alph_rmse_train=np.inf
        min_alph_rmse_validation=np.inf
        for i in range(-6,-2):
            rmse_array_validation=[]
            rmse_array_training=[]
            for k_fold in range (0,k_folds): 
                #print("{},{},{}".format(l,i,k_fold))
                X_test=X[:k_fold_test_rows]
                y_test=y[:k_fold_test_rows]
                
                X_train=X[k_fold_test_rows:t_rows]
                y_train=y[k_fold_test_rows:t_rows]
                
                w=gradient_descent_ridge_reg(X_train,y_train,k_fold_train_rows,t_cols,10**i,2**l)
                #print(w)
                rms1=find_rmse_gradient_descent(X_test,y_test,w)
                rms2=find_rmse_gradient_descent(X_train,y_train,w)
                if(np.isfinite(rms1)):
                    rmse_array_validation.append(rms1)
                    #print(rmse_array_validation)
                if(np.isfinite(rms2)):
                    rmse_array_training.append(rms2)
                    #print(rmse_array_training)
                X_new=np.concatenate((X_train,X_test))
                y_new=np.concatenate((y_train,y_test))
                X=np.copy(X_new)
                y=np.copy(y_new)
            avg_rmse_training=np.average(rmse_array_training)
            avg_rmse_validation=np.average(rmse_array_validation)  
            #print(avg_rmse_validation)
            if(avg_rmse_training<=min_alph_rmse_train):
                min_alph_rmse_train=avg_rmse_training
            if(avg_rmse_validation<=min_alph_rmse_validation):
                min_alph_rmse_validation=avg_rmse_validation
        rms_train.append(min_alph_rmse_train)
        rms_val.append(min_alph_rmse_validation)
    print("rmse train {}".format(rms_train))
    print("rmse validation {}".format(rms_val))
    lambda_arr=[2**-16,2**-15,2**-14,2**-13,2**-12,2**-11,2**-10,2**-9,2**-8,2**-7,2**-6,2**-5,2**-4,2**-3,2**-2,2**-1]
    fig, ax = plt.subplots()
    #ax2 = ax.twinx()
    ax.plot(lambda_arr,rms_train,marker='o',label='train')
    ax.plot(lambda_arr,rms_val,marker='*',label='validation')
    ax.legend(loc='best')
    plt.show()

getdata();