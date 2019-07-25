# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:22:23 2019

@author: IIST
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
def centralize_data(result):
    mean_vector=np.ndarray.mean(result,axis=0)
    result2=np.subtract(result,mean_vector)
    return result2,mean_vector
def sigmoid(x):
    return 1/(1+np.exp(-x))

def getdata(): 
    reader = csv.reader(open("Data2.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    print(result.shape)
    t_rows,t_cols=result.shape
    data=result[1:t_rows,1:t_cols]
    t_rows,t_cols=data.shape
    data,mean_vector=centralize_data(data)
    print(data.shape)
    W=np.identity(t_cols)
    error=1e-5
    alpha=1e-3
    norm=1
    count=0
    while(error<norm and count<500):
        count=count+1
        for i in range(0,t_rows):
            temp=np.dot(W,data[i].T)
            temp2=1-2*sigmoid(temp)
            temp3=np.outer(temp2,data[i])
            Wt=(W.T)
            inverse = np.linalg.inv(Wt)
            Wnew=W+alpha*(temp3+inverse)
            w_difference=np.subtract(Wnew,W)
            norm=np.linalg.norm(w_difference)
            W=np.copy(Wnew)
        #print(norm)
    #np.savetxt("W_ICA.csv", W, delimiter=",")   
    S=np.dot(data,W)
    plt.figure()
    for i in range(3):        
        plt.plot(S[:,i])
    plt.figure()
    for i in range(3):        
        plt.plot(data[:,i])
    np.savetxt("sources.csv", S, delimiter=",")   
getdata()