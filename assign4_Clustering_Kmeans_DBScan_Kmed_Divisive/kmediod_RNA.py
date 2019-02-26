# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 21:51:19 2018

@author: IIST
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits import mplot3d
from sympy.plotting import plot3d
from scipy.spatial import distance

def random_color(p):
    COLORS=['r', 'g', 'b', 'k', 'y', 'm', 'c','chartreuse','burlywood']
    return COLORS[p]

def plotdata(result,k,cluster):
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                plt.scatter(result[i,0],result[i,1],c=random_color(p+3))
                
def plot3data(result,k,cluster):
    #fig = plt.figure()
    ax = plt.axes(projection='3d')
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                ax.scatter(result[i,0],result[i,1],result[i,2],c=random_color(p+3))

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

def k_means(data,u_array,k):
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

def euclidean(data,u_array):
    return np.linalg.norm(np.subtract(data,u_array)) 

def distance_bet_pts(result,datapt):
    med_distance=np.linalg.norm(np.subtract(result,datapt))
    return med_distance
                
def getdata(): 
    reader = csv.reader(open("data3.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    min_DB=np.inf
    for k in range(2,6):
        clus,mean_arr=k_mediod(result,k)
        DB=find_db_index(result,clus,mean_arr)
        if(DB<min_DB):
            min_DB=DB
            best_k=k
            best_cluster=clus
    print(best_k)
    plot3data(result,best_k,best_cluster)
    np.savetxt('cluster_data3_kmediod.csv',best_cluster)
    
def k_mediod(result,k):
    idx = np.arange(result.shape[0])
    selected = np.random.choice(idx, k, replace=False)
    
    med_array=[]
    for i in range(len(selected)):        
        med_array.append(result[selected[i]])      

    cluster=k_means(result,med_array,k)
    iterator=0
    i=0
    while(1):             
        cluster_i=np.where(cluster==i)
        l1=np.ndarray.tolist(cluster_i[0])       
        xij=[]
        for p in l1:
            xij.append(result[p])  
        med_distance=distance_bet_pts(xij,med_array[i])
        min_dist=np.inf
        for p in l1:
            p_distance=distance_bet_pts(xij,result[p])
            if(p_distance<min_dist):
                min_dist=p_distance
                min_index=p       
        #print(min_index)
        mediod_old=np.copy(med_array)
        if(min_dist<med_distance):
            med_array[i]=result[min_index]
            cluster=k_means(result,med_array,k)
        i=(i+1)%k
        flag=[]
        for array in range(len(mediod_old)): 
            med_new=np.copy(med_array)
            mediod_old=np.array(mediod_old)
            med_new=np.array(med_new)
            if(np.array_equal(med_new[array],mediod_old[array])):
                flag.append(True)
            else:
                flag.append(False)
        #print(flag)
        if(all(element for element in flag)):
            break
        iterator=iterator+1
    return cluster,med_new

getdata()