# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 23:56:57 2019

@author: IIST
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:39:44 2019

@author: IIST
"""
import csv
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
#from sklearn.metrics import davies_bouldin_score
def plotdata(result,k,cluster):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(k)]      
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                plt.scatter(result[i,0],result[i,1],c=color[p]) 

def getdata(): 
    reader = csv.reader(open("data2.csv", "rt"), delimiter=",")
    x = list(reader)
    data = np.array(x).astype("float")
    t_rows,t_cols=data.shape
    k=15
    neighbors = NearestNeighbors(k, algorithm='auto').fit(data)
    distance, indices = neighbors.kneighbors(data)
    indices=np.delete(indices,0,1)
    distance=np.delete(distance,0,1)
    
    W=np.zeros((data.shape[0],data.shape[0]))
    
    for i in range(data.shape[0]):
        index=indices[i]
        for j in range(index.shape[0]):
            val=indices[index[j]]
            if (np.isin(i, val)):
                 W[i,index[j]]=distance[i,j]
                 W[index[j],i]=distance[i,j]
                 
    D=np.diag([np.sum(row) for row in W])
    L=D-W
    d= np.diagonal(D)
    d= np.power(d,-1/2)
    D = np.diag(d)        
    L_prime = np.dot(D,L)
    L=np.dot(L_prime,D)
    k=3
    w, v = np.linalg.eigh(L)
    print(w)
    vector=v[:,:k]
    vector_normed=normalize(vector, norm='l2', axis=1)    
    value=w[:k]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(vector_normed)
    cluster=kmeans.labels_
    plotdata(data,k,cluster)
    plt.figure()
    plt.plot(w)
    np.savetxt('data1_eigenvalues_normalized.csv',value,delimiter=",")
    np.savetxt('data1_eigenvectors_normalized.csv',vector_normed,delimiter=",")
    
    
            
getdata()    