# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:27:15 2018

@author: IIST
"""

import csv
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

#import random

def random_color(p):
    COLORS=['r', 'g', 'b', 'k', 'y', 'm', 'c','chartreuse','burlywood']
    return COLORS[p]

def k_means(data,u_array,k):
    dist_from_mean=np.linalg.norm(np.subtract(data,u_array[0]),axis=1)   
    print(dist_from_mean)
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
        
def getdata(): 
    reader = csv.reader(open("data2_o.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    rows,cols=result.shape
    k=3
    idx = np.arange(result.shape[0])
    selected = np.random.choice(idx, k, replace=False)
    
    u_array=[]
    for i in range(len(selected)):        
        u_array.append(result[selected[i]])      
    #cluster=k_means(result,u_array,k)
    while(1):
        cluster=k_means(result,u_array,k)
        u_array2=[]
        count=0
        for i in range(k):
            cluster_i=np.where(cluster==i)
            l1=np.ndarray.tolist(cluster_i[0])
            sum_arr=np.zeros(result.shape[1])
            for j in l1:
                sum_arr=sum_arr+result[j]
            u_array2.append(sum_arr/len(l1))   
            if(np.array_equal(u_array[i],u_array2[i])):
                count=count+1
        if(count==k):
            break;
        u_array=np.copy(u_array2)           
        print(u_array)    
    plotdata(result,k,cluster) 
        
def plotdata(result,k,cluster):
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                plt.scatter(result[i,0],result[i,1],c=random_color(p))
         
getdata()