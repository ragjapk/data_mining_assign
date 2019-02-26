# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:01:40 2018

@author: IIST
"""
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from performance_calculator import find_performance_measures_NB
import pandas as pd
from performance_calculator import get_final_performance

import math
def getdata(): 
    reader = csv.reader(open("adultdata.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    rows,cols=result.shape  
    X=np.delete(result,cols-1,1)
    y=np.delete(result, np.s_[0:cols-1], axis=1)            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)
   
    mean=np.zeros((X_train.shape[1],1),dtype=float)  
    
    std_dev=np.zeros((X_train.shape[1],1),dtype=float)
    for i in range(1,X_train.shape[1]):
        mean[i]=np.mean(X_train[:,i])
        std_dev[i]=np.std(X_train[:,i])
        
    for i in range(1,X_train.shape[1]):
        X_train[:,i]=(np.subtract(X_train[:,i],mean[i]))/std_dev[i]
        
    for i in range(1,X_test.shape[1]):
        X_test[:,i]=np.subtract(X_test[:,i],mean[i])/std_dev[i]
    gda2(X_train,y_train,X_test,y_test)
    
def normal_dbn_probability(X,u,sigma,sigma_inverse,no_attributes):
    temp1=np.subtract(X,u)
    temp2=np.dot(np.dot(temp1.T,sigma_inverse),temp1)
    exponent=math.exp(-temp2)
    constant=((2*math.pi)**(no_attributes/2))*np.linalg.det(sigma)
    return((1/constant)*exponent)     
    
def gda2(X_train,y_train,X_test,y_test):
    print(X_train.shape[0])
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

    y_pred=np.zeros((X_test.shape[0]))
    for i in range(X_test.shape[0]):
        p_x_given_y_1=normal_dbn_probability(X_test[i,:],u1,cov,cov_i,X_test.shape[1])
        p_x_given_y_0=normal_dbn_probability(X_test[i,:],u0,cov,cov_i,X_test.shape[1])
        
        prior_prob_1=len(u1_array)/(X_train.shape[0])
        prior_prob_0=len(u0_array)/(X_train.shape[0])   

            
        numerator1=prior_prob_1*p_x_given_y_1
        numerator2=prior_prob_0*p_x_given_y_0
        
        p_y_1_given_x=numerator1/(numerator1+numerator2)
        
        
        p_y_0_given_x=numerator2/(numerator1+numerator2)
        
        if p_y_1_given_x>p_y_0_given_x:
            y_pred[i]=1
        else:
            y_pred[i]=0 
    tp,fp,fn,tn=find_performance_measures_NB(y_test,y_pred)
    sensitivity,specificity,precision,accuracy,f_measure=get_final_performance(tp,fp,tn,fn)
    print("Accuracy is {}".format(accuracy))
    print("Sensitivity is {}".format(sensitivity))
    print("Specificity is {}".format(specificity))
    print("Precision is {}".format(precision))
    print("F_Measure is {}".format(f_measure))
getdata()