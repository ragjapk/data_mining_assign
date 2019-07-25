# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:33:11 2019

@author: IIST
"""
import numpy as np
import csv
def create_kernel_matrix(kernel_val,X,Y):
    if(kernel_val=='L'):
        k=X@Y.T
    elif(kernel_val=='P'):
        k=np.power(X@Y.T,2)
    elif(kernel_val=='E'):
        sigma =40
        k= np.exp((-1/(2*sigma**2))*np.linalg.norm(X[:,None]-Y, axis=2))
    elif(kernel_val=='H'):
        c=10
        k = 1/np.sqrt(np.linalg.norm(X[:,None]-Y, axis=2)+c)
    return k

def centralise(K):
    ones=np.full((K.shape[0],K.shape[0]),1/K.shape[0])
    K1=np.dot(K,ones)
    K2=np.dot(ones,K)
    K3=ones@K@ones
    Knew=K-K1-K2+K3
    return Knew

def getdata():
    reader = csv.reader(open("data5.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    K=create_kernel_matrix('P',X,X)
    Knew=centralise(K)
    eigval,alpha=np.linalg.eigh(Knew)
    
    idx = eigval.argsort()[::-1]   
    eigenValues = eigval[idx]
    eigenVectors = alpha[:,idx]
    
    eigSum=np.sum(eigenValues)
    k=0
    sum=0
    for i in range(1,len(eigenValues)):
        sum=sum+eigenValues[i-1]       
        error=sum/eigSum
        #print(error)
        if(error>=0.95):
            k=i
            break      
    #for i in range(len(eigval)):
        #if(eigval[i]>1e-3):
            #break 
    #eigval_new=eigval[i:]
    #alpha_new=alpha[:,i:]    
    alpha_new=eigenVectors[:,:k] 
    eigval_new=eigenValues[:k]
    print(len(eigval_new))
    alpha_new=alpha_new.T
    m=0
    for row in alpha_new:
        row=row/np.sqrt(eigval_new[m])
        m=m+1
    Xnew=np.zeros((X.shape[0],m))
    for i in range(X.shape[0]):
        for j in range(m):
            Xnew[i,j]=np.dot(alpha_new[j],K[i])
    print(Xnew.shape)
    print(y.shape)
    y=y.reshape(-1,1)
    final=np.hstack((Xnew,y))
    np.savetxt('data5_kpca_poly.csv',final,delimiter=',')

getdata()
            