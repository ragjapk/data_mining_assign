# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:01:40 2018

@author: IIST
"""
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from performance_calculator import find_performance_measures_gda
import pandas as pd
from performance_calculator import get_final_performance

import math
def getdata(): 
    reader = csv.reader(open("wdbcdata.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    rows,cols=result.shape  
    X=np.delete(result,cols-1,1)
    y=np.delete(result, np.s_[0:cols-1], axis=1)             
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)    
    gda2(X_train,y_train,X_test,y_test)
     
def gda2(X_train,y_train,X_test,y_test):
    #print(X_train.shape[0])
    cov=np.zeros((X_train.shape[1],X_train.shape[1]),dtype=float)
    u1_array=[]
    u0_array=[]
    positive=0
    u0=np.zeros((X_train.shape[1],1),dtype=float) 
    u1=np.zeros((X_train.shape[1],1),dtype=float)     
    positive=0
    for j in range(0,X_train.shape[0]):
        if y_train[j]==1 :
            u1_array.append(X_train[j,:])
            positive=positive+1
        else:
            u0_array.append(X_train[j,:]) 
    u1_array=np.array(u1_array)
    u0_array=np.array(u0_array)
    u1=np.ndarray.mean(u1_array,axis=0)
    u0=np.ndarray.mean(u0_array,axis=0)
    cov=np.cov(X_train.T)
    cov_i=np.linalg.inv(cov) 
    temp=math.log(positive/(X_train.shape[0]-positive))
    temp2=u1.T.dot(cov_i).dot(u1)     
    temp3=u0.T.dot(cov_i).dot(u0)   
    w0=temp-0.5*temp2+0.5*temp3
    w0=np.array(w0)
    wt=np.dot(cov_i,np.subtract(u1,u0))   
    w_final= np.insert(wt, 0, w0, axis=0)
    print(w_final)
    np.savetxt("sample_w.csv", w_final, delimiter=",")   
    tp,fp,fn,tn=find_performance_measures_gda(X_test,y_test,wt,w0)
    #print(tp,fp,fn,tn)
    sensitivity,specificity,precision,accuracy,f_measure=get_final_performance(tp,fp,tn,fn)
    print("Accuracy is {}".format(accuracy))
    print("Sensitivity is {}".format(sensitivity))
    print("Specificity is {}".format(specificity))
    print("Precision is {}".format(precision))
    print("F_Measure is {}".format(f_measure))
getdata()