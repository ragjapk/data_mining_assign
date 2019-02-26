# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 21:32:08 2018

@author: IIST
"""

import csv
import numpy as np
from kmeans import k_means_algo
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
def random_color(p):
    COLORS=['r', 'g', 'b', 'k', 'y', 'm', 'c','chartreuse','burlywood']
    return COLORS[p]

def get_cmap(n, name='rgb'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plotdata(result,k,cluster):
    #cmap = get_cmap(k)
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):                
                plt.scatter(result[i,0],result[i,1],c=random_color(p+5))

def distance_bet_pts(result,datapt):
    med_distance=np.linalg.norm(np.subtract(result,datapt))
    return med_distance


def find_most_dissimilar(cluster,result):
    xij=[]
    cluster=np.array(cluster)
    unique1, counts1 = np.unique(cluster, return_counts=True)
    dist_arr=[]
    for i in unique1:
        #print(i)
        dist=0
        cluster_i=np.where(cluster==i)
        l1=np.ndarray.tolist(cluster_i[0])  
        #print(len(l1))
        for p in l1:
                xij.append(result[p])  
        for p in l1:
            dist=dist+distance_bet_pts(xij,result[p])
        dist_arr.append(dist)
    dist_arr=np.array(dist_arr)
    
    return np.argmax(dist_arr)  

def getdata(): 
    #reader = csv.reader(open("IRIS.csv", "rt"), delimiter=",")
    #x = list(reader)
    df = pd.read_csv("data.csv")
    df=df.iloc[:,1:]
    result=df.values
    #result = np.array(result).astype("float")
    k=3
    rows,cols=result.shape
    cluster=k_means_algo(result,2) 
    iteration=2
    while(iteration<k):
        index=find_most_dissimilar(cluster,result) 
        print(index)
        cluster__=np.where(cluster==index)
        l1_=np.ndarray.tolist(cluster__[0]) 
        xij_=[]
        indices=[]
        for p in l1_:
                xij=np.reshape(result[p,:],(1,result[p,:].shape[0]))
                #dtpt2=np.transpose(result[p])
                xij_.append(xij)
                indices.append(p)
        xij__=np.vstack(elem for elem in xij_)        
        cluster_=k_means_algo(xij__,2)
        for p in l1_:
            for m in range(len(cluster)):            
                if(m==p):
                    if(cluster_[indices.index(p)]==0):                    
                        cluster[m]=index
                    else:
                        cluster[m]=iteration
                    break
        iteration=iteration+1
    #plotdata(result,k,cluster)
    #print(cluster)
    np.savetxt('cluster_rna_divisive.csv',cluster)
    unique1, counts1 = np.unique(cluster, return_counts=True)
    print(unique1,counts1)
    
    #print(l2)
    #while():
def plot3data(result,k,cluster):
    #fig = plt.figure()
    ax = plt.axes(projection='3d')
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                ax.scatter(result[i,0],result[i,1],result[i,2],c=random_color(p+5))        
           
getdata()