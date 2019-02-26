# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:03:15 2018

@author: IIST
"""
import numpy as np
import math

def find_performance_measures(X_test,y_test,w):
    temp=np.zeros((X_test.shape[0],1),dtype=float)  
    y_pred=np.zeros((y_test.shape[0],1),dtype=float)  
    for i in range(X_test.shape[0]):
                temp[i]=np.dot(w.T,X_test[i])
                if temp[i]>0:
                    y_pred[i]=1
                else:
                    y_pred[i]=0
    Tp,Fp,Fn,Tn=0,0,0,0
    for i in range(len(y_test)):
        if y_test[i]==1 and y_pred[i]==1:
            Tp=Tp+1
        elif y_test[i]==0 and y_pred[i]==0:
            Tn=Tn+1
        elif y_test[i]==1 and y_pred[i]==0:
            Fn=Fn+1
        else:
            Fp=Fp+1
    return Tp,Fp,Fn,Tn

def find_performance_measures_data2(X_test,y_test,w):
    temp=np.zeros((X_test.shape[0],1),dtype=float)  
    y_pred=np.zeros((y_test.shape[0],1),dtype=float)  
    for i in range(X_test.shape[0]):
                temp[i]=np.dot(w.T,X_test[i])
                if temp[i]>0:
                    y_pred[i]=1
                else:
                    y_pred[i]=-1
    Tp,Fp,Fn,Tn=0,0,0,0
    for i in range(len(y_test)):
        if y_test[i]==1 and y_pred[i]==1:
            Tp=Tp+1
        elif y_test[i]==-1 and y_pred[i]==-1:
            Tn=Tn+1
        elif y_test[i]==1 and y_pred[i]==-1:
            Fn=Fn+1
        else:
            Fp=Fp+1
    return Tp,Fp,Fn,Tn

def find_performance_measures_gda(X_test,y_test,w,w0):
    temp=np.zeros((X_test.shape[0],1),dtype=float)  
    y_pred=np.zeros((y_test.shape[0],1),dtype=float)  
    for i in range(X_test.shape[0]):
                temp[i]=np.dot(w.T,X_test[i])
                temp[i]=temp[i]+w0
                temp[i]=1/(1+math.exp(-temp[i]))
                if temp[i]>=0.5:
                    y_pred[i]=1
                else:
                    y_pred[i]=0
    Tp,Fp,Fn,Tn=0,0,0,0
    for i in range(len(y_test)):
        if y_test[i]==1 and y_pred[i]==1:
            Tp=Tp+1
        elif y_test[i]==0 and y_pred[i]==0:
            Tn=Tn+1
        elif y_test[i]==1 and y_pred[i]==0:
            Fn=Fn+1
        else:
            Fp=Fp+1
    return Tp,Fp,Fn,Tn

def find_performance_measures_roc(prob,y_test,threshold): 
    y_pred=np.zeros((y_test.shape[0],1),dtype=float)
    for i in range(y_test.shape[0]):
        if prob[i]>=threshold:
            y_pred[i]=1
        else:
            y_pred[i]=0
    Tp,Fp,Fn,Tn=0,0,0,0
    for i in range(len(y_test)):
        if y_test[i]==1 and y_pred[i]==1:
            Tp=Tp+1
        elif y_test[i]==0 and y_pred[i]==0:
            Tn=Tn+1
        elif y_test[i]==1 and y_pred[i]==0:
            Fn=Fn+1
        else:
            Fp=Fp+1
    sensitivity=Tp/(Tp+Fn)
    specificity=Tn/(Tn+Fp)
    return sensitivity,1-specificity,y_pred

def find_performance_measures_roc_data2(prob,y_test,threshold): 
    y_pred=np.zeros((y_test.shape[0],1),dtype=float)
    for i in range(y_test.shape[0]):
        if prob[i]>=threshold:
            y_pred[i]=1
        else:
            y_pred[i]=-1
    Tp,Fp,Fn,Tn=0,0,0,0
    for i in range(len(y_test)):
        if y_test[i]==1 and y_pred[i]==1:
            Tp=Tp+1
        elif y_test[i]==-1 and y_pred[i]==-1:
            Tn=Tn+1
        elif y_test[i]==1 and y_pred[i]==-1:
            Fn=Fn+1
        else:
            Fp=Fp+1
    sensitivity=Tp/(Tp+Fn)
    specificity=Tn/(Tn+Fp)
    return sensitivity,1-specificity,y_pred

def get_threshold(X_test,y_test,w):
    temp=np.dot(w.T,X_test)
    temp=1/(1+math.exp(-temp))
    return temp

def find_performance_measures_NB(y_test,y_pred):
    Tp,Fp,Fn,Tn=0,0,0,0
    for i in range(len(y_test)):
        if y_test[i]==1 and y_pred[i]==1:
            Tp=Tp+1
        elif y_test[i]==0 and y_pred[i]==0:
            Tn=Tn+1
        elif y_test[i]==1 and y_pred[i]==0:
            Fn=Fn+1
        else:
            Fp=Fp+1
    return Tp,Fp,Fn,Tn

def find_performance_measures_NB_old(y_test,y_pred):
    Tp,Fp,Fn,Tn=0,0,0,0
    for i in range(len(y_test)):
        if y_test[i]==2 and y_pred[i]==2:
            Tp=Tp+1
        elif y_test[i]==4 and y_pred[i]==4:
            Tn=Tn+1
        elif y_test[i]==2 and y_pred[i]==4:
            Fn=Fn+1
        else:
            Fp=Fp+1
    return Tp,Fp,Fn,Tn

def get_final_performance(tp,fp,tn,fn):
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    precision=tp/(tp+fp)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    f_measure=(2*precision*sensitivity)/(sensitivity+precision)
    return sensitivity,specificity,precision,accuracy,f_measure

def find_rmse_gradient_descent(X_test,y_test,w):
    sum=0
    f_x=np.dot(X_test,w)
    for i in range(0,len(X_test)):
        sum=sum+((f_x[i]-y_test[i])**2)
        
    rmse=math.sqrt(sum/len(X_test))
    return rmse