# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 00:38:42 2018

@author: IIST
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

def random_color(p):
    COLORS=['r', 'g', 'b', 'k', 'y', 'm', 'c']
    return COLORS[p]

def plotdata(result,k,cluster):
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                plt.scatter(result[i,0],result[i,1],c=random_color(p+1))

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
    reader = csv.reader(open("data2_o.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    k=3
    idx = np.arange(result.shape[0])
    selected = np.random.choice(idx, k, replace=False)
    
    med_array=[]
    for i in range(len(selected)):        
        med_array.append(result[selected[i]])      

    cluster=k_means(result,med_array,k)
    iterator=0
    i=0
    flag=[]
    while(iterator<2):             
        cluster_i=np.where(cluster==i)
        l1=np.ndarray.tolist(cluster_i[0])
        p_distance=np.zeros((len(l1),1))
        index_store=np.zeros((len(l1),1),dtype='uint8')
        j=0
        xij=[]
        for p in l1:
            xij.append(result[p])  
        med_distance=distance_bet_pts(xij,med_array[i])
        for p in l1:
            p_distance[j]=distance_bet_pts(xij,result[p])
            index_store[j]=p
            print(p)
            j=j+1
        #print(i,len(l1))
        min_distance=np.amin(p_distance)
        min_index=np.argmin(p_distance)
        print(min_index)
        #print(min_distance,med_distance)
        mediod_old=np.copy(med_array)
        #print(mediod_old)
        if(min_distance<med_distance):
            ind=index_store[min_index]
            print(ind)
            med_array[i]=result[ind]
            cluster=k_means(result,med_array,k)
        i=(i+1)%k
        #print(med_array,len(med_array))
        #print(mediod_old,len(mediod_old))
        for array in range(len(mediod_old)): 
            if(np.array_equal(med_array[array],mediod_old[array])):
                #print(med_array[array],mediod_old[array])
                flag.append(True)
            else:
                flag.append(False)
        #print(flag)
        if(all(element for element in flag)):
            break
        iterator=iterator+1
    plt.figure()
    plotdata(result,k,cluster)  

getdata()