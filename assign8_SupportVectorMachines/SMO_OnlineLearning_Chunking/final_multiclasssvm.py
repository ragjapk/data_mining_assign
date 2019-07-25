# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:06:55 2019

@author: IIST
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 23:48:10 2019

@author: IIST
"""
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import hyperparam_kernel as SVM
from sklearn.metrics import accuracy_score
def getdata():
    dataset = pd.read_csv('Iris.csv',skiprows=1)
    y=dataset.iloc[:,-1]
    le = LabelEncoder().fit(y)
    y_new=le.transform(y)
    y_new=np.asanyarray(y_new,dtype=np.double)
    classes, counts = np.unique(y_new, return_counts=True)
    #classes = list(le.classes_)     
    x = dataset.iloc[:, :-1]
    X=np.array(x).astype("float")
    KPCA(X,y,classes,y)

def KPCA(X,y_new,classes,y):
    X_train,X_test,y_train,y_test =train_test_split(X,y_new,stratify=y)
    rows=X_test.shape[0]
    cols=len(classes)
    classifiers=np.zeros((rows,cols))
    for i in range(len(classes)):
        y_train1=np.copy(y_train)
        y_train1[y_train ==classes[i]]=1
        y_train1[y_train !=classes[i]]=-1
        #print(type(y_train1))
        y_test1=np.copy(y_test)
        y_test1[y_test ==classes[i]]=1
        y_test1[y_test !=classes[i]]=-1
        bestc,bestk=SVM.kfold_validate(X_train,y_train1)
        score,sol,s_v,f,bias,alphas,prediction=SVM.svm(X_train,y_train1,X_test,y_test1,bestc,bestk)
        prediction=prediction.reshape(-1,38)
        classifiers[:,i]=prediction
    predicted = np.argmax(classifiers,axis=1)
    print(predicted)
    accuracy = accuracy_score(y_test, predicted)   
    print(accuracy)
    #x = list(reader)
    #H = np.array(x).astype("float")
    #X=np.delete(H,-1,axis=1)
    #y=np.take(H, -1, axis=1)
 
getdata()