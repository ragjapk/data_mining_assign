# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:21:02 2018

@author: IIST
"""

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
from gradient_descent import gd_ridge_jw
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    mean=np.zeros((X_train.shape[1],1),dtype=float)  
    
    std_dev=np.zeros((X_train.shape[1],1),dtype=float)
    for i in range(1,X_train.shape[1]):
        mean[i]=np.mean(X_train[:,i])
        std_dev[i]=np.std(X_train[:,i])
        
    for i in range(1,X_train.shape[1]):
        X_train[:,i]=(np.subtract(X_train[:,i],mean[i]))/std_dev[i]
    for i in range(1,X_test.shape[1]):
        X_test[:,i]=np.subtract(X_test[:,i],mean[i])/std_dev[i]
    holdout(X_train,X_test,y_train,y_test,t_cols)
    
def holdout(X_train,X_test,y_train,y_test,t_cols):
    jw=gd_ridge_jw(X_train,y_train,X_train.shape[0],t_cols,10**-3,2**-5)   
    #np.arange(0,jw.shape[0],1)
    plt.plot(jw)


getdata();