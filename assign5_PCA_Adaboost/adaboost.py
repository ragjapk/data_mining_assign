# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 12:48:04 2019

@author: IIST
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from random import choices
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
def bootstrap_sample(X,y,D):
    a=list(range(0, X.shape[0]))
    a=np.array(a)
    dnew=np.ndarray.tolist(D)
    train_indices = choices(a,weights=dnew,k= X.shape[0])
    bxtrain=X[np.unique(train_indices)]
    bytrain=y[np.unique(train_indices)]
    return bxtrain,bytrain,np.unique(train_indices)

def classifier(X,y,t):    
    
    classifiers=[]
    logistic = LogisticRegression()
    knn = KNeighborsClassifier(n_neighbors=3)
    gnb = GaussianNB()
    classifiers.append(logistic)
    classifiers.append(knn)
    #classifiers.append(dtree)
    classifiers.append(gnb)
    #classifiers=list(logistic,knn,dtree,gnb)
    model=classifiers[t].fit(X, y.ravel())
    ypred=model.predict(X)
    #print(model.score(X, y))
    return ypred   
   
    
T=3    
alpha=np.zeros((T))
#dataset = np.array(list(csv.reader(open("data.csv", "rt"), delimiter=","))).astype('float'
df = pd.read_csv('adaboost_dataset.csv',header=None)
df.replace('?', np.nan, inplace= True)
df = df.fillna(method='ffill')
data2=df.values
dataset = np.array(data2, dtype='float')
row,col=dataset.shape
print(row)
X=np.delete(dataset,col-1,axis=1)
y=np.take(dataset, col-1, axis=1)
np.place(y, y==0, -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)
D=np.zeros((X_train.shape[0]),dtype='float')
D.fill((1/X_train.shape[0]))
hht=[]
for t in range(0,T):
    #Dividing to boot strap samples
    bxtrain,bytrain,train_indices=bootstrap_sample(X_train,y_train,D)
    #Getting predictions from ith classifier
    ht=classifier(bxtrain,bytrain,t)
    boolean=np.not_equal(ht,bytrain)
    #Finding those indices in the data where the mis classification happened
    indices=np.where(boolean)
    D_t=D[train_indices]
    et=np.sum(D[indices]) 
    #If total error is less than 0.5, set weights for the particular classifier
    #Update weights of training points that were mis classified
    if(et<=0.5):
        alpha[t]=0.5*np.log((1-et)/et)
        A= D_t*np.exp(-1*alpha[t]*ht)
        D_t = A/np.sum(A)
        D[train_indices]=D_t
    else:
        t=t-1
        continue
ht_final=np.zeros(X_test.shape[0])
for t in range(0,T):
    ht=classifier(X_test,y_test,t) 
    ht_final=ht_final+(alpha[t]*ht)
ht_final=np.sign(ht_final)
    
score=accuracy_score(y_test, ht_final, normalize=True, sample_weight=None)
print(score)    

print(classification_report(y_test, ht_final))