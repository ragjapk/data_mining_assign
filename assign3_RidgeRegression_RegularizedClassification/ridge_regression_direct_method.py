# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:11:59 2018

@author: IIST
"""

import csv
import numpy as np
import math
import time
from sklearn.model_selection import train_test_split

start = time.time()
reader = csv.reader(open("data1.csv", "rt"), delimiter=",")
x = list(reader)
result = np.array(x).astype("float")

t_rows,t_cols=result.shape

#Normalization
t_rows,t_cols=result.shape
X=np.delete(result,t_cols-1,1)
y=np.delete(result, np.s_[0:t_cols-1], axis=1)
X=np.insert(X,0,1,axis=1)
#Normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
mean=np.zeros((X_train.shape[1],1),dtype=float)  

std_dev=np.zeros((X_train.shape[1],1),dtype=float)
for i in range(1,X_train.shape[1]):
    mean[i]=np.mean(X_train[:,i])
    std_dev[i]=np.std(X_train[:,i])
    
for i in range(1,X_train.shape[1]):
    X_train[:,i]=(np.subtract(X_train[:,i],mean[i]))/std_dev[i]
for i in range(1,X_test.shape[1]):
    X_test[:,i]=np.subtract(X_test[:,i],mean[i])/std_dev[i]

for i in range(-25,0):
    lambdaa=2**i
    Xt=np.transpose(X_train)
    XtX=np.dot(Xt,X_train)
    XtX=XtX+lambdaa*np.identity(XtX.shape[0])
    inverse = np.linalg.inv(XtX)
    Xty=np.dot(Xt,y_train)
    w=np.dot(inverse,Xty)
    sum=0.0
    f_xi=np.dot(X_test,w)
    for i in range(0,X_test.shape[0]):
        sum=sum+((f_xi[i]-y_test[i])**2)
        
    rmse=math.sqrt(sum/X_test.shape[0])
    print("validation RMSEs")
    print(rmse)
    print("train RMSEs")
    f_xi=np.dot(X_train,w)
    for i in range(0,X_test.shape[0]):
        sum=sum+((f_xi[i]-y_train[i])**2)
    rmse2=math.sqrt(sum/X_train.shape[0])
    print(rmse2)
#np.savetxt("w_direct_method.csv", w, delimiter=",")
end = time.time()
print("Running time is {}".format(end-start))