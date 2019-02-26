# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:59:23 2018

@author: IIST
"""
import numpy as np

def gradient_ascent_ridge_reg(X_train,y_train,train_rows,t_cols,alpha,lambdaa):    
    epsilon=0.0001
    w=np.zeros((t_cols,1), dtype=float)
    w_new=np.zeros((t_cols,1), dtype=float)
    w_difference=np.zeros((t_cols,1), dtype=float)
    f_xi=np.zeros((train_rows,1),dtype=float)  
    temp=np.zeros((train_rows,1),dtype=float) 
    difference_vector=np.zeros((train_rows,1),dtype=float)   
    w_norm=10
    wt=np.transpose(w)
    m=0
    while (w_norm>epsilon and m<700):   
        for i in range(0,train_rows):    
            temp[i]=np.dot(wt,X_train[i]) 
        f_xi=1/(1+np.exp(-temp))
        difference_vector=np.subtract(y_train,f_xi)   
        w_new[0]=w[0]+alpha*(np.sum(difference_vector))  
        diff_transpose=np.transpose(difference_vector)
            
        for j in range(1,t_cols):
           k=np.dot(diff_transpose,X_train[:,j])
           w_new[j]=w[j]*(1-alpha*lambdaa)+alpha*k
       
        w_difference=np.subtract(w_new,w)
        w=np.copy(w_new)
        wt=np.transpose(w)
        w_norm=np.linalg.norm(w_difference)
        print(w_norm)
        m=m+1
    return w