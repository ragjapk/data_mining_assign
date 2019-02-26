# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:01:40 2018

@author: IIST
"""
import numpy as np
import pandas as pd
from performance_calculator import find_performance_measures_NB
from sklearn.model_selection import train_test_split
from performance_calculator import get_final_performance
def discretize_attribute(df,y,no_of_bins):
    df[y]=pd.qcut(df[y],no_of_bins,duplicates='drop')
    df[y] = df[y].apply(lambda x: x.mid)

def getdata(): 
    df = pd.read_csv('adultdata.csv',header=None)
    #df = df.drop([0],axis=1)
    #df.replace('?', np.nan, inplace= True)
    #df = df.fillna(method='ffill')
    #Adult Data Set
    discretize_attribute(df,0,7)
    discretize_attribute(df,2,100)
    #discretize_attribute(df,4,40)
    discretize_attribute(df,10,100)
    discretize_attribute(df,11,50)
    discretize_attribute(df,12,6)
    
    #BC Data Set
    #for i in range(30):
        #discretize_attribute(df,i,7)
    
    result=df.values
    
    X=np.delete(result,result.shape[1]-1,1)
    y=np.delete(result, np.s_[0:result.shape[1]-1], axis=1) 
    n=5
    accuracy=0
    sensitivity=0
    specificity=0
    precision=0
    f_measure=0
    for i in range(n):             
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)
        acc,sen,spe,pre,f_m=naive_bayes(X_train,y_train,X_test,y_test)
        accuracy=accuracy+acc
        sensitivity=sensitivity+sen
        specificity=specificity+spe
        precision=precision+pre
        f_measure=f_measure+f_m
    print("Accuracy is {}".format(accuracy/n))
    print("Sensitivity is {}".format(sensitivity/n))
    print("Specificity is {}".format(specificity/n))
    print("Precision is {}".format(precision/n))
    print("F_Measure is {}".format(f_measure/n))
     
def naive_bayes(X_train,y_train,X_test,y_test):
    y_pred=np.zeros((y_test.shape[0]),dtype=int)
    unique, counts = np.unique(y_train, return_counts=True)
    y_count=dict(zip(unique, counts))
    positive=y_count[1]
    negative=y_count[0]   
    num_positive=[]
    num_negative=[]  
    for i in range(X_train.shape[0]):
        if y_train[i]==1:
            num_positive.append(X_train[i])
        if y_train[i]==0:
            num_negative.append(X_train[i])
    num_positive=np.array(num_positive)
    num_negative=np.array(num_negative)    
    x_given_y_1=[]
    x_given_y_0=[] 
    for i in range(num_positive.shape[1]):        
        unique1, counts1 = np.unique(num_positive[:,i], return_counts=True)
        unique, counts = np.unique(num_negative[:,i], return_counts=True)
        #print(unique1,counts1)
        #print(unique,counts)  
        for j in range(len(counts1)):
            counts1[j]=counts1[j]+1
        for j in range(len(counts)):
            counts[j]=counts[j]+1
          
        x_count_positive=dict(zip(unique1, counts1))  
        x_count_negative=dict(zip(unique, counts))    
        diff = set(x_count_negative.keys()) ^ set(x_count_positive.keys())
        for k in diff:
                if k not in x_count_positive:
                    x_count_positive[k]=1
                else:
                    x_count_negative[k]=1
        x_given_y_1.append(x_count_positive)
        x_given_y_0.append(x_count_negative)
   
    for i in range(X_test.shape[0]):
        probability_1=1
        probability_0=1           
        for j in range(X_test.shape[1]):
            key=float(X_test[i,j]) 
            
            dictionary=x_given_y_1[j] 
            if key in dictionary:          
                num=dictionary[key]
                den=(positive+len(dictionary))
            else:
                num=1
                den=(positive+len(dictionary)+1)
            #print("Positive:{}".format(num/den))    
            probability_1=probability_1*(num/den)            
            dictionary=x_given_y_0[j]
            if key in dictionary:          
                num=dictionary[key]
                den=(negative+len(dictionary))
            else:
                num=1
                den=(negative+len(dictionary)+1)
            probability_0=probability_0*(num/den)
            #print("Negative:{}".format(num/den)) 
        numerator=(positive/X_train.shape[0])*probability_1
        p_y_1_given_x=numerator
        
        numerator=(negative/X_train.shape[0])*probability_0
        p_y_0_given_x=numerator
        
        #print(p_y_1_given_x,p_y_0_given_x)
        
        if p_y_1_given_x>p_y_0_given_x:
            y_pred[i]=1
        else:
            y_pred[i]=0
    tp,fp,fn,tn=find_performance_measures_NB(y_test,y_pred)
    accuracy,sensitivity,specificity,precision,f_measure=get_final_performance(tp,fp,tn,fn)
    return accuracy,sensitivity,specificity,precision,f_measure
    

getdata()