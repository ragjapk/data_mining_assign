# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:36:32 2019

@author: IIST
"""

import numpy as np
import csv
from numpy import linalg as LA
import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def centralize_data(result):
    mean_vector=np.ndarray.mean(result,axis=0)
    result2=np.subtract(result,mean_vector)
    return result2,mean_vector

def pca(data,k):
    w, v = LA.eigh(np.dot(data.T,data))
    j=0
    V=np.zeros((len(w), k))
    W=np.zeros((k))
    for i in range(len(w)-1,len(w)-k-1,-1):
        V[:,j]=v[:,i]
        W[j]=w[i]
        j=j+1
    result3=np.dot(data,V) 
    dim=np.arange(1,len(w)+1,1)
    plt.plot(dim,w)
    return result3,V,W

def reconstruction(data,mean,V,k):
    datanew=np.zeros((data.shape[0],V.shape[0]))
    '''for i in range(0,data.shape[0]):
        summ=np.zeros((V.shape[0]))
        for j in range(0,k):
            scalar=np.dot(data[i],V[:,j])
            summ=summ+scalar*V[:,j]
        datanew[i]=summ'''
    datanew=np.dot(data,V.T)
    result_final=np.add(datanew,mean)    
    return result_final

'''s = open('ad.csv','r').read()
chars = ('?')#('$','%','?','^','*') # etc
for c in chars:
   s = '0'.join( s.split(c) )
out_file = open('pre_processed_data.csv','w')
out_file.write(s)
out_file.close()'''    
reader = csv.reader(open("pre_processed_data_.csv", "rt"), delimiter=",")
x = list(reader)
k=6
dataset = np.array(x).astype("float")
result=np.delete(dataset,0,axis=1)
y=np.take(dataset, 0, axis=1)

reg = GaussianNB()
reg.fit(result, y.ravel())
print('original datas score {}'.format(reg.score(result, y)))

result2,mean=centralize_data(result)
mindiff=np.inf

result3,V,W=pca(result2,k)
#reg = GaussianNB()
#reg.fit(result3, y.ravel())
#print('score for {} datas score {}'.format(k,reg.score(result3, y)))
print(k)
result4=reconstruction(result3,mean,V,k)
diff=np.linalg.norm(result4-result)
if(mindiff>diff):
    mindiff=diff
    mink=k
print('Best k is {}'.format(k))
    

