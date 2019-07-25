# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:34:39 2019

@author: IIST
"""
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
b=0
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

def create_kernel_function(kernel_val,X,Y):
    if(kernel_val=='L'):
        k=X@Y.T
    elif(kernel_val=='P'):
        k=np.power(X@Y.T,2)
    elif(kernel_val=='E'):
        sigma =40
        k= np.exp((-1/(2*sigma**2))*np.linalg.norm(X-Y))
    elif(kernel_val=='H'):
        c=10
        k = 1/np.sqrt(np.linalg.norm(X-Y)+c)
    return k

def takeStep(c,X,y,alphas,i1,i2,E1,kernel_val):
    global b
    eps=1e-3
    if (i1 == i2):
        return 0
    alpha1 =alphas[i1]
    y1 = y[i1]
    alpha2 =alphas[i2]
    y2 = y[i2]
    E2 =  get_predicted(alphas,X,y,i2,kernel_val) - y2
    s = y1*y2
    if y1 == y2 :
        L = max(0, alpha2 + alpha1 - c)
        H = min(c, alpha2 + alpha1)
    else :
        L = max(0, alpha2 - alpha1)
        H = min(c, c + alpha2 - alpha1)
    if L == H :
        return 0
    k11 = create_kernel_function(kernel_val,X[i1],X[i1])
    k12 = create_kernel_function(kernel_val,X[i1],X[i2])
    k22 = create_kernel_function(kernel_val,X[i2],X[i2])
    eta = k11+k22-2*k12
    if(eta > 0):
        a2 = alpha2 + y2*(E1-E2)/eta
        if (a2 < L):
            a2 = L
        elif (a2 > H):
            a2 = H
    else:
        a2=L
        f1=y1*(E1+b)-alpha1*k11-s*a2*k12
        f2=y2*(E2+b)-s*alpha1*k12-a2*k22
        L1=alpha1+s*(a2-L)
        H1=alpha1+s*(a2-H)
        Lobj = L1*f1+L*f2+0.5*(L1**2)*k11+0.5*(L**2)*k22+s*L*L1*k12
        Hobj = H1*f1+H*f2+0.5*(H1**2)*k11+0.5*(H**2)*k22+s*H*H1*k12
        if (Lobj < Hobj-eps):
            a2 = L
        elif (Lobj > Hobj+eps):
            a2 = H
        else:
            a2 = alpha2
    if (np.abs(a2-alpha2) < eps*(a2+alpha2+eps)):
        return 0
    a1=alpha1+s*(alpha2-a2)
    b1=E1+y1*(alpha1-a1)*k11+y2*(alpha2-a2)*k12+b    
    b2=E2+y1*(alpha1-a1)*k12+y2*(alpha2-a2)*k22+b
    if (a1 != 0 and a1 != c):
        b = b1    
    elif (a2 != 0 and a2 != c):
        b = b2
    elif (L != H):
        b = (b1 + b2) / 2
    alphas[i1] = a1
    alphas[i2] = a2
    return 1

def get_predicted(alpha,X,y,i,kernel_val) :
    global b
    K1=create_kernel_matrix(kernel_val,X,X[i])
    alpha=alpha.reshape(alpha.shape[0],1)
    y=y.reshape(y.shape[0],1)
    alpha_y=alpha*y
    f=np.dot(alpha_y.T,K1)
    #f=f.reshape(f.shape[1],1)
    f_tilda=f-b
    prediction=np.sign(f_tilda) 
    return prediction

def get_nonzero_c_alpha(c,alpha):
    sv = np.where( np.logical_and( alpha > 0, alpha < c) )[0] 
    return sv

def second_choice_heuristic(X,y,alphas,c,E2,kernel_val):
    global b
    errorcache = np.zeros(y.shape)
    for i in range(y.shape[0]) :
        errorcache[i] = (get_predicted(alphas,X,y,i,kernel_val) - b) - y[i] 
    alpha_index = get_nonzero_c_alpha(c,alphas)
    error = errorcache[alpha_index] - E2
    max_index = np.argmax(abs(error))
    return max_index

def examineExample(X,y,alphas,C,i1,kernel_val):
    tol=1e-3
    y1 = y[i1]
    alpha1 = alphas[i1]
    E1 = get_predicted(alphas,X,y,i1,kernel_val) - y1
    r1 = E1*y1
    sv_index=get_nonzero_c_alpha(C,alphas)
    temp_alpha=alphas[sv_index]
    if ((r1 < -tol and alpha1 < C) or (r1 > tol and alpha1 > 0)):
        if (len(sv_index) > 1):
            i2 = second_choice_heuristic(X,y,alphas,C,E1,kernel_val)
            if (takeStep(C,X,y,alphas,i1,i2,E1,kernel_val)):
                return 1
        rand_index = np.random.permutation(len(temp_alpha))
        for i2 in rand_index:
            if takeStep(C,X,y,alphas,i1,i2,E1,kernel_val):
                return 1
        rand_index = np.random.permutation(len(X))
        for i2 in rand_index:
            if takeStep(C,X,y,alphas,i1,i2,E1,kernel_val):
                return 1
    return 0


def smo(X,y,c,kernel_val):
    global b
    #b=0
    alphas = np.zeros(y.shape)
    numChanged = 0
    examineAll = 1
    while (numChanged > 0 or examineAll):
        numChanged = 0
        if (examineAll):
            for i in range(X.shape[0]):
                numChanged += examineExample(X,y,alphas,c,i,kernel_val)
        else:
            sv_index = get_nonzero_c_alpha(c,alphas)
            for i in sv_index:
                numChanged += examineExample(X,y,alphas,c,i,kernel_val)
        if (examineAll == 1):
            examineAll = 0
        elif (numChanged == 0):
            examineAll = 1
    return alphas

def predict_output(alphas,X_train,y_train,kernel_,X_test,y_test):
    global b
    index=np.where(alphas>0)[0]
    alpha_sv = alphas[index]
    X_sv = X_train[index]
    y_sv = y_train[index]
    
    y_sv=y_sv.reshape(y_sv.shape[0],1)
    alpha_sv=alpha_sv.reshape(alpha_sv.shape[0],1)
    alpha_y= alpha_sv*y_sv    
    #Creation of Kernel Matrix for Testing
    K=create_kernel_matrix(kernel_,X_sv,X_test)    
    f=np.dot(alpha_y.T,K)
    f=f.reshape(f.shape[1],1)
    f_tilda=f+b
    prediction=np.sign(f_tilda)  
    score=accuracy_score(y_test, prediction, sample_weight=None)
    return score
 
def getdata():
    reader = csv.reader(open("data4.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)
    best_c,bestkernel=kfold_validate(X_train,y_train)
    alphas=smo(X_train,y_train,best_c,bestkernel)
    accuracy=predict_output(alphas,X_train,y_train,bestkernel,X_test,y_test)
    print('Accuracy obtained is {}'.format(accuracy))
    print('C obtained after cross validation is {}'.format(best_c))
    print('Best kernel is {}'.format(bestkernel))
    #plot_data_with_labels(X_train,y_train,X_test,y_test,best_c,alphas,bias,bestkernel,sv)
    
    #
def kfold_validate(X,y):
    skf = KFold(n_splits=5)
    skf.get_n_splits(X, y)
    max_acc=np.NINF
    for kernel in ['H','E','L','P']:
        for e in range(3,7,2):
            accuracy=[]
            c=10**e
            print(c,kernel)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]            
                alphas=smo(X_train,y_train,c,kernel)
                acc=predict_output(alphas,X_train,y_train,kernel,X_test,y_test)
                accuracy.append(acc)
            accuracy=np.asarray(accuracy)
            avg_acc=np.average(accuracy)
            print(avg_acc)
            if(avg_acc>max_acc):
                max_acc=avg_acc
                best_c=c
                bestkernel=kernel
    return best_c,bestkernel  
           
getdata()