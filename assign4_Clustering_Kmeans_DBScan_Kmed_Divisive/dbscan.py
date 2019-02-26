# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 18:42:25 2018

@author: IIST
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial import distance
import random
from mpl_toolkits import mplot3d
import pandas as pd

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
                
def core_object(p,min_pts,eps_nbhd,index_matrix,dist_matrix):
    pts=0
    dist_array=index_matrix[p]
    eps_array=[]
    for i in dist_array:
        if(dist_matrix[p,i]<eps_nbhd):
            pts=pts+1
            eps_array.append(i)
    if(pts>=min_pts):
        return True, eps_array
    else:
        return False, eps_array
    
def getdata(): 
    reader = csv.reader(open("data4.csv", "rt"), delimiter=",")
    x = list(reader)
    #df = pd.read_csv("data4.csv", header=None, sep=';')
    #df.drop('reports', axis=1)
    #df=df.iloc[:,1:]
    #result=df.values
    result = np.array(x).astype("float")
    
      
    dist_matrix = distance.cdist(result, result, 'euclidean')
    index_matrix=np.argsort(dist_matrix)
    index_matrix=np.delete(index_matrix,0, axis=1) 
    cluster=np.zeros((result.shape[0],1))
    pts=result.shape[0]
    visited=[True]*pts
    min_pts=3
    eps_nbhd=0.92
    #print(result)   
    iteration=0
    while(1):
        if((not(any(element for element in visited)))):
            break
        result_sub=np.where(visited)
        choice=random.choice(result_sub[0])
        visited[choice]=False
        is_core,N=core_object(choice,min_pts,eps_nbhd,index_matrix,dist_matrix)
        if(is_core):
            iteration=iteration+1
            cluster[choice]=iteration
            for pt in N:
                if(visited[pt]):
                    visited[pt]=False
                    is_core,N_=core_object(pt,min_pts,eps_nbhd,index_matrix,dist_matrix)
                    if(is_core):
                        N.extend(N_)
                if(cluster[pt]==0):
                    cluster[pt]=iteration        
    cluster=np.array(cluster)
    unique1, counts1 = np.unique(cluster, return_counts=True)
    print(unique1)
    #print(cluster)
    plotdata(result,len(unique1),cluster)
    
getdata()