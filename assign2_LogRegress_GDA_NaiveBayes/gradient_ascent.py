# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:31:34 2018

@author: IIST
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:26:39 2018

@author: IIST
"""
import numpy as np

def gradient_ascent(X_train,y_train,train_rows,t_cols,alpha):    
    epsilon=0.001
    #w=np.random.random((t_cols,1))
    #w = np.float_(w)
    w=np.zeros((t_cols,1), dtype=float)
    w_new=np.zeros((t_cols,1), dtype=float)
    w_difference=np.zeros((t_cols,1), dtype=float)
    f_xi=np.zeros((train_rows,1),dtype=float)  
    temp=np.zeros((train_rows,1),dtype=float)  
    difference_vector=np.zeros((train_rows,1),dtype=float)   
    w_norm=10
    wt=np.transpose(w)
    while w_norm>epsilon:   
        for i in range(0,train_rows):    
            temp[i]=np.dot(wt,X_train[i])            #
        f_xi=1/(1+np.exp(-temp))
        difference_vector=np.subtract(y_train,f_xi)    
        diff_transpose=np.transpose(difference_vector)            
        for j in range(0,t_cols):
           k=np.dot(diff_transpose,X_train[:,j])
           w_new[j]=w[j]+ alpha*k
           
        w_difference=np.subtract(w_new,w)
        w=np.copy(w_new)
        wt=np.transpose(w)
        w_norm=np.linalg.norm(w_difference)
        #print(w_norm)
    #print(w)
    return w


def gradient_ascent_iterative(X_train,y_train,train_rows,t_cols,alpha):    
    w=np.zeros((t_cols,1), dtype=float)
    w_new=np.zeros((t_cols,1), dtype=float)
    f_xi=np.zeros((train_rows,1),dtype=float)  
    temp=np.zeros((train_rows,1),dtype=float)  
    difference_vector=np.zeros((train_rows,1),dtype=float)   
    wt=np.transpose(w)
    for i in range(0,150):
        for i in range(0,train_rows):    
            temp[i]=np.dot(wt,X_train[i])            #
        f_xi=1/(1+np.exp(-temp))
        difference_vector=np.subtract(y_train,f_xi)    
        diff_transpose=np.transpose(difference_vector)            
        for j in range(0,t_cols):
           k=np.dot(diff_transpose,X_train[:,j])
           w_new[j]=w[j]+ alpha*k
        w=np.copy(w_new)
        wt=np.transpose(w)
    return w


