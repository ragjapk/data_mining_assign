# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:27:15 2018

@author: IIST
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#import random
import pandas as pd
def random_color(p):
    COLORS=['r', 'g', 'b', 'k', 'y', 'm', 'c']
    return COLORS[p]

def plot3data(result,k,cluster):
    #fig = plt.figure()
    ax = plt.axes(projection='3d')
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                ax.scatter(result[i,0],result[i,1],result[i,2],c=random_color(p))

def cluster_up(data,u_array,k):
    dist_from_mean=np.linalg.norm(np.subtract(data,u_array[0]),axis=1)   
    #print(dist_from_mean)
    cluster=np.zeros((data.shape[0],1))
    for i in range(1,k):
        #print(i)
        new_dist=np.linalg.norm(np.subtract(data,u_array[i]),axis=1) 
        boolean=np.greater(dist_from_mean,new_dist) 
        index_where_greater=np.where(boolean)
        for j in index_where_greater:
            cluster[j]=i
            dist_from_mean[j]=new_dist[j]
    return cluster
    #for i in range(result.shape[0]):
def euclidean(data,u_array):
    return np.linalg.norm(np.subtract(data,u_array)) 
        
def getdata(k): 
    reader = csv.reader(open("q4.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    #df = pd.read_csv("data.csv")
    #df=df.iloc[:,1:]
    #result=df.values
    rows,cols=result.shape
    clus=k_means_algo(result,k)
    print(clus)
def k_means_algo(result,k) : 
    idx = np.arange(result.shape[0])
    selected = np.random.choice(idx, k, replace=False)
    u_array=[]
    for i in range(len(selected)):  
        u_array.append(result[selected[i]])  
    Jcu_arr=[]
    iterator=0
    while(1):
        iterator=iterator+1
        J_c_u=0
        cluster=cluster_up(result,u_array,k)
        u_array2=[]
        count=0
        for i in range(k):
            cluster_i=np.where(cluster==i)
            l1=np.ndarray.tolist(cluster_i[0])
            sum_arr=np.zeros(result.shape[1])
            xij=[]
            for j in l1:
                sum_arr=sum_arr+result[j]
                xij.append(result[j])
            u_array2.append(sum_arr/len(l1))
            for it in range(len(xij)):
                J_c_u=J_c_u+euclidean(xij[it],sum_arr/len(l1))
            
            if(np.array_equal(u_array[i],u_array2[i])):
                count=count+1
        Jcu_arr.append(J_c_u)    
        if(count==k):
            break;
        u_array=np.copy(u_array2)           
    iterator=np.arange(0,iterator,1)    
    #plt.figure()
    plotdata(result,k,cluster) 
    #plt.figure()
    #plt.scatter(iterator,Jcu_arr)    
    return cluster
def plotdata(result,k,cluster):
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                plt.scatter(result[i,0],result[i,1],c=random_color(p))
         
getdata(3)