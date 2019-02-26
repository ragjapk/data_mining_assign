# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:16:47 2018

@author: IIST
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
import pandas as pd
from scipy.spatial import distance

def plotdata(result,k,cluster):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(k)]  
    
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                plt.scatter(result[i,0],result[i,1],c=color[p])

def plot3data(result,k,cluster):
    #fig = plt.figure()
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(k)]  
    ax = plt.axes(projection='3d')
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                ax.scatter(result[i,0],result[i,1],result[i,2],c=color[p]) 



def cluster_up(data,u_array,k):
    dist_from_mean=np.linalg.norm(np.subtract(data,u_array[0]),axis=1)   
    cluster=np.zeros((data.shape[0],1))
    for i in range(1,k):
        new_dist=np.linalg.norm(np.subtract(data,u_array[i]),axis=1) 
        boolean=np.greater(dist_from_mean,new_dist) 
        index_where_greater=np.where(boolean)
        for j in index_where_greater:
            cluster[j]=i
            dist_from_mean[j]=new_dist[j]
    return cluster

def euclidean(data,u_array):
    return np.linalg.norm(np.subtract(data,u_array)) 

def find_db_index(result,clus,mean_arr):
    unique, counts = np.unique(clus, return_counts=True)
    S=[]
    n_cluster=len(unique)
    j=0
    for i in (unique):
        cluster_i=np.where(clus==i)
        l1=np.ndarray.tolist(cluster_i[0])        
        subset=np.take(result,l1,axis=0)
        dist=euclidean(subset,mean_arr[j])
        S.append(dist/counts[j])
        j=j+1
    dij = distance.cdist(mean_arr, mean_arr, 'euclidean')
    R=[]
    Ri_max=[]
    for i in range(len(unique)-1):
        Ri=np.NINF
        for j in range(i+1,len(unique)):
            Rij=(S[i]+S[j])/dij[i,j]
            R.append(Rij)
            if(Rij>Ri):
                Ri=Rij
        Ri_max.append(Ri)   
        
    DB=(sum(Ri_max))/n_cluster
    print(DB)
    return DB
        
def getdata(): 
    reader = csv.reader(open("q4.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    #df = pd.read_csv("data.csv")
    #df=df.iloc[:,1:]
    #result=df.values
    rows,cols=result.shape
    min_DB=np.inf
    for k in range(2,5):
        clus,mean_arr=k_means_algo(result,k)
        DB=find_db_index(result,clus,mean_arr)
        if(DB<min_DB):
            min_DB=DB
            best_k=k
            best_cluster=clus
    print(best_k)
    plotdata(result,best_k,best_cluster)
    np.savetxt('cluster_q4_kmeans.csv',best_cluster)
    
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
        u_array=np.copy(u_array2)     
        if(count==k):
            break;
              
    iterator=np.arange(0,iterator,1)     
    return cluster,u_array

         
getdata()