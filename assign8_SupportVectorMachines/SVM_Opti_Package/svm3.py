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
    return k

COLORS=['red','blue']

def plot_separator(ax, w, b): 
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    x = np.arange(0, 10)
    ax.plot(x, x * slope + intercept, 'k-')
    
def plot_data_with_labels(x, y, ax):
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        ax.scatter(x_sub[:, 0], x_sub[:, 1], c = COLORS[li])
    
def fit(X,y,K,c):
    NUM = X.shape[0]
    P = cvxopt.matrix(K)
    q = cvxopt.matrix(-np.ones((NUM, 1)))
    G = cvxopt.matrix(-np.eye(NUM))
    h = cvxopt.matrix(np.zeros(NUM))
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(np.zeros(1))
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    alphas[alphas<0]=0
    alphas[alphas>c]=c
    return alphas

def getdata():
    reader = csv.reader(open("data1.csv", "rt"), delimiter=",")
    x = list(reader)
    H = np.array(x).astype("float")
    X=np.delete(H,-1,axis=1)
    y=np.take(H, -1, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)
    best_c=kfold_validate(X_train,y_train)
    accuracy=svm(X_train,y_train,X_test,y_test,best_c)
    print(accuracy)
    print(best_c)
    
    #
def kfold_validate(X,y):
    e=5
    c=10**e
    return c

            
def svm(X_train,y_train,X_test,y_test,c):    
    kernel_='L'
    #Creation of inital Kernel Matrix
    K=create_kernel_matrix(kernel_,X_train,X_train)
    Y=np.outer(y_train,y_train)
    K=K*Y
    
    #Get Alphas.
    alphas=fit(X_train,y_train,K,c)
    y_train=y_train.reshape(y_train.shape[0],1)
    alpha_y= alphas*y_train
    f_train=np.dot(alpha_y.T,K)
    print(f_train.shape)
    diff=y_train-f_train.T
    cnt = ((alphas >0) & (alphas <c)).shape[0]
    bias=np.sum(diff)/cnt    
    print(bias)
    f=np.zeros((X_test.shape[0]))   
    
    #Creation of Kernel Matrix for Testing
    K=create_kernel_matrix(kernel_,X_train,X_test)
    Y=np.outer(y_train,y_test)
    K1=K*Y
    
    f=np.dot(alpha_y.T,K1)
    f=f.reshape(f.shape[1],1)
    f_tilda=f+bias
    prediction=np.sign(f_tilda)  
    score=accuracy_score(y_test, prediction, sample_weight=None)
    print(score)
    return score

    
getdata()