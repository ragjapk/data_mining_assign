
import numpy as np
import csv
from gradient_ascent import gradient_ascent
from performance_calculator import find_performance_measures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from performance_calculator import get_final_performance
import time
def getdata(): 
    reader = csv.reader(open("data1.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    rows,cols=result.shape
    X=np.delete(result,cols-1,1)
    y=np.delete(result, np.s_[0:cols-1], axis=1)
    X=np.insert(X,0,1,axis=1)                
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y,)

    mean=np.zeros((X_train.shape[1],1),dtype=float)  
    
    std_dev=np.zeros((X_train.shape[1],1),dtype=float)
    for i in range(1,X_train.shape[1]):
        mean[i]=np.mean(X_train[:,i])
        std_dev[i]=np.std(X_train[:,i])
        
    for i in range(1,X_train.shape[1]):
        X_train[:,i]=(np.subtract(X_train[:,i],mean[i]))/std_dev[i]
        
    alpha,acc=k_fold(X_train,y_train)
    start=time.time()
    w=gradient_ascent(X_train,y_train,X_train.shape[0],X_train.shape[1],alpha)
    end=time.time()
    print(end-start)
    #Normalize X_test:
    for i in range(1,X_test.shape[1]):
        X_test[:,i]=np.subtract(X_test[:,i],mean[i])/std_dev[i]
    np.savetxt("w_logistic_reg_data1.csv", w, delimiter=",")       
    tp,fp,fn,tn=find_performance_measures(X_test,y_test,w)
    sensitivity,specificity,precision,accuracy,f_measure=get_final_performance(tp,fp,tn,fn)
    print("Accuracy on test data is {}".format(accuracy))
    print("Sensitivity on test data is {}".format(sensitivity))
    print("Specificity on test data is {}".format(specificity))
    print("Precision on test data is {}".format(precision))
    print("F_Measure on test data is {}".format(f_measure))
    
def k_fold(X,y):
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, y)
    best_alpha=0
    avg_accuracy=0
    max_accuracy=0
    alpha_list=[0.000001,0.00001,0.0001,0.001,0.01,0.1] 
    best_alpha=alpha_list[1]
    itera=0
    #alpha_list=[0.0001] 
    for alpha in alpha_list:   
        accuracy_array=[]
        iteration=0
        for train_index, test_index in skf.split(X, y):
            iteration=iteration+1
            #print("TRAIN:", train_index, "TEST:", test_index)        
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]                                   
            w=gradient_ascent(X_train,y_train,X_train.shape[0],X_train.shape[1],alpha)
            tp,fp,fn,tn=find_performance_measures(X_test,y_test,w)
            sensitivity,specificity,precision,accuracy,f_measure=get_final_performance(tp,fp,tn,fn)
            accuracy_array.append(f_measure)
            print(itera,iteration)
        avg_accuracy=np.average(accuracy_array)
        if(avg_accuracy>max_accuracy):
            max_accuracy=avg_accuracy
            best_alpha=alpha
        itera=itera+1
    print(best_alpha)
    return (best_alpha,max_accuracy)


getdata()  

