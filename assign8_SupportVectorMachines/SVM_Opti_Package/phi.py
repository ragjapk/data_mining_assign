# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:43:19 2019

@author: IIST
"""
import numpy as np
import math
import csv
H=[[1,1,-1],[1,-1,1],[-1,-1,-1],[-1,1,1]]

def phi(X):
    Xnew=np.zeros((X.shape[0],4))
    Xnew[:,0]=np.square(X[:,0])
    Xnew[:,1]=np.square(X[:,1])
    Xnew[:,2]=np.multiply(X[:,0],X[:,1])
    Xnew[:,2]=Xnew[:,2]*math.sqrt(2)
    return Xnew
  
reader = csv.reader(open("H.csv", "rt"), delimiter=",")
x = list(reader)
H = np.array(x).astype("float")
X=np.delete(H,-1,axis=1)
y=np.take(H, -1, axis=1)
Xnew=phi(X)
Xnew[:,3]=y
np.savetxt('H_new.csv',Xnew,delimiter=',')
    