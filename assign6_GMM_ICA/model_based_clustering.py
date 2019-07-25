# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:16:43 2019

@author: IIST
"""
import csv
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import metrics
#from sklearn.metrics import davies_bouldin_score
def getdata(): 
    reader = csv.reader(open("Data1.csv", "rt"), delimiter=",")
    x = list(reader)
    data = np.array(x).astype("float")
    t_rows,t_cols=data.shape
    maxsc=np.NINF
    for k in range(2,5):
        print(k)
        sigmas = [np.identity(t_cols) for _ in range(k)]
        w=np.ones(k)*1/k
        #ui=np.random.randint(min(data[:,0]),max(data[:,0]),size=(k,len(data[0])))
        ui = np.random.random(size=(k,len(data[0])))
        sc,clus,w,sigmas,u=EM_Algo(data,k,t_rows,t_cols,sigmas,w,ui)
        print(sc)
        if sc>maxsc:
            maxsc=sc
            best_clus=clus
            bestk=k
            bestw=w
            bestsig=sigmas
            bestu=u
    print('Silhouette Coeff is:{} and best k is: {}'.format(maxsc,bestk))
    print(bestsig)
    print(bestu)
    print(bestw)
    plotdata(data,bestk,best_clus)
    
def normal_dbn_probability(X,u,sigma,n):
    sigma_inverse=np.linalg.inv(sigma)
    temp1=np.subtract(X,u)
    temp2=np.dot(np.dot(temp1.T,sigma_inverse),temp1)
    exponent=math.exp(-temp2)
    constant=((2*math.pi)**(n/2))*np.linalg.det(sigma)
    return((1/constant)*exponent) 
    
def plotdata(result,k,cluster):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(k)]      
    for p in range(k):
        for i in range(0,len(result)):
            if(cluster[i]==p):
                plt.scatter(result[i,0],result[i,1],c=color[p]) 
                
def EM_Algo(X,k,N,n,sigmas,w,u):
    count=0
    P_xj_ci=np.zeros((N,k),dtype='float')
    Pij=np.zeros((N,k),dtype='float')
    while(count<110):
        count=count+1        
        for i in range(k):
            for j in range(N):
                P_xj_ci[j,i]=normal_dbn_probability(X[j],u[i],sigmas[i],n)
        for i in range(k):
            Pij[:,i]=np.divide(w[i]*P_xj_ci[:,i],(np.dot(w,P_xj_ci.T).T))
        for i in range(k):
            sum1=np.zeros((n))
            for j in range(N):           
                sum1=sum1+(X[j]*Pij[j,i])
            Pij_col_sum=np.sum(Pij[:,i],axis=0)
            u[i]=sum1/Pij_col_sum
            sum_cov=np.zeros((n,n))
            for j in range(N):  
                sum_cov=sum_cov+np.outer(X[j]-u[i],X[j]-u[i])*Pij[j,i]             
            sigmas[i]=sum_cov/Pij_col_sum
            w[i]=Pij_col_sum/N
    clusters=np.argmax(Pij,axis=1)
    sc=metrics.silhouette_score(X, clusters, metric='euclidean')
    return sc,clusters,w,sigmas,u        
getdata()