# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 23:30:09 2019

@author: IIST
"""
import numpy as np
import csv
import cvxopt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def create_kernel_matrix(kernel_val,X,Y):
    if(kernel_val=='L'):
        k=X@Y.T
    elif(kernel_val=='P'):
        k=np.power(X@Y.T,2)
    elif(kernel_val=='E'):
        sigma =100
        k= np.exp((-1/(2*sigma**2))*np.linalg.norm(X[:,None]-Y, axis=2))
    elif(kernel_val=='H'):
        c=10
        k = 1/np.sqrt(np.linalg.norm(X[:,None]-Y, axis=2)+c)
    return k

    
def plot_data_with_labels(X_train,y_train,X_test,y_test,c,alphas,bias,kernel,sv):
    x_min=X_test[:, 0].min() - 1
    x_max =X_test[:, 0].max() + 1
    y_min = X_test[:, 1].min() - 1
    y_max = X_test[:, 1].max() + 1
    h = 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    X_p = np.c_[xx.ravel(), yy.ravel()]
    pred=test_svm(X_train,y_train,X_p,c,alphas,bias,kernel)
    y_p= pred.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx,yy,y_p,cmap='Accent',alpha=0.7)
    X_test_p = X_test[y_test == 1]
    X_test_n = X_test[y_test == -1]
    plt.scatter(X_test_p[:,0],X_test_p[:,1],c='blue')
    plt.scatter(X_test_n[:,0],X_test_n[:,1],c='red')
    plt.scatter(sv[:,0],sv[:,1],c='green')
    plt.show()
    
def fit(X,y,K,c):
    #print(y.shape)
    NUM = X.shape[0]
    P = cvxopt.matrix(K)
    q = cvxopt.matrix(-np.ones((NUM, 1)))
    G = cvxopt.matrix(-np.eye(NUM))
    h = cvxopt.matrix(np.zeros(NUM))
    A = cvxopt.matrix(y.reshape(1, -1))
    #print(A.size)
    b = cvxopt.matrix(np.zeros(1))
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    alphas[alphas<0]=0
    alphas[alphas>c]=c
    return alphas,sol

def getdata():
    reader = csv.reader(open("data5_kpca_hyperbolic.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)
    best_c,bestkernel=kfold_validate(X_train,y_train)
    accuracy,sol,sv,f,bias,alphas,pred=svm(X_train,y_train,X_test,y_test,best_c,bestkernel)
    print('Duality GAP is: {}'.format(sol['gap']))
    print('Dual Objective Function value is: {}'.format(sol['primal objective']))
    print('Primal Objective Function value is: {}'.format(sol['dual objective']))
    print('Dual Slack variable value is {}'.format(sol['primal slack']))
    print('Primal Slack variable value is {}'.format(sol['dual slack']))
    print('Value of bias is {}'.format(bias))
    print('Accuracy obtained is {}'.format(accuracy))
    print('C obtained after cross validation is {}'.format(best_c))
    print('Best kernel is {}'.format(bestkernel))
    #supp_X=X_train[sv]
    #supp_y=y_train[sv]        
    #np.savetxt('supportvectors.csv',supp_X,delimiter=',')
    #np.savetxt('supportvectors_labels.csv',supp_y,delimiter=',')
    #plot_data_with_labels(X_train,y_train,X_test,y_test,best_c,alphas,bias,bestkernel,sv)
    
    #
def kfold_validate(X,y):
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X, y)
    max_acc=np.NINF
    for kernel in ['H']:
        for e in range(-4,6,2):
            accuracy=[]
            c=10**e
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]            
                acc,sol,sv,f,bias,alphas,pred=svm(X_train,y_train,X_test,y_test,c,kernel)
                accuracy.append(acc)
            accuracy=np.asarray(accuracy)
            avg_acc=np.average(accuracy)
            #print(avg_acc)
            if(avg_acc>max_acc):
                max_acc=avg_acc
                best_c=c
                bestkernel=kernel
    return best_c,bestkernel  
            
def svm(X_train,y_train,X_test,y_test,c,kernel_):    
    #Creation of inital Kernel Matrix
    M = np.arange(0, X_train.shape[0],1)
    K=create_kernel_matrix(kernel_,X_train,X_train)
    Y=np.outer(y_train,y_train)
    K1=K*Y    
    #Get Alphas.
    alphas,sol=fit(X_train,y_train,K1,c)
    M=M.reshape(-1,1)
    sv = M[alphas>0]    
    y_train=y_train.reshape(y_train.shape[0],1)
    alpha_y= alphas*y_train
    f_train=np.dot(alpha_y.T,K)
    diff=y_train-f_train.T
    cnt = ((alphas >0) & (alphas <c)).shape[0]
    bias=np.sum(diff)/cnt    
    f=np.zeros((X_test.shape[0]))   
    
    #Creation of Kernel Matrix for Testing
    K1=create_kernel_matrix(kernel_,X_train,X_test)
    
    f=np.dot(alpha_y.T,K1)
    f=f.reshape(f.shape[1],1)
    f_tilda=f+bias
    prediction=np.sign(f_tilda)  
    score=accuracy_score(y_test, prediction, sample_weight=None)
    return score,sol,sv,f,bias,alphas,prediction

def test_svm(X_train,y_train,X_test,c,alphas,bias,kernel):    
    y_train=y_train.reshape(y_train.shape[0],1)
    alpha_y= alphas*y_train    
    #Creation of Kernel Matrix for Testing
    K=create_kernel_matrix(kernel,X_train,X_test)    
    f=np.dot(alpha_y.T,K)
    f=f.reshape(f.shape[1],1)
    f_tilda=f+bias
    prediction=np.sign(f_tilda)  
    return prediction

#getdata()