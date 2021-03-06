# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:35:06 2019

@author: IIST
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:25:12 2019

@author: IIST
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:36:32 2019

@author: IIST
"""
from sklearn.linear_model import LinearRegression
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

def pca(data):
    eigenValues, eigenVectors = LA.eigh(np.dot(data.T,data))
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    eigSum=np.sum(eigenValues)
    sum=0
    k=0
    for i in range(1,data.shape[1]+1):
        sum=sum+eigenValues[i-1]       
        error=sum/eigSum
        #print(error)
        if(error>=0.9999):
            k=i
            break   
    v2=eigenVectors[:,:k]    
    result3=np.dot(data,eigenVectors[:,:k]) 
    dim=np.arange(1,data.shape[1]+1,1)
    #plt.plot(dim,eigenValues)
    return result3,k,eigenVectors[:,:k]

def reconstruction(data,mean,V,k):
    #datanew=np.zeros((data.shape[0],V.shape[0]))
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
reader = csv.reader(open("slice_localization_data.csv", "rt"), delimiter=",")
x = list(reader)
dataset = np.array(x).astype("float")
row,col=dataset.shape
result=np.delete(dataset, col-1,axis=1)
y=np.take(dataset,  col-1, axis=1)

reg = LinearRegression()
reg.fit(result, y.ravel())
print('original datas score {}'.format(reg.score(result, y)))

result2,mean=centralize_data(result)
mindiff=np.inf

result3,k,V=pca(result2)
np.savetxt("slice_localization_pca.csv", result3, delimiter=",")  
reg = LinearRegression()
reg.fit(result3, y.ravel())
print('score for {} datas score {}'.format(k,reg.score(result3, y)))
result4=reconstruction(result3,mean,V,k)
reg.fit(result4, y.ravel())
print('score for final datas score {}'.format(reg.score(result4, y)))
np.savetxt("slice_localization_pca_regression.csv", result4, delimiter=",") 
np.savetxt("slice_localization_pca_optimalbasis.csv", V, delimiter=",") 
diff=np.linalg.norm(result4-result)
print('Best k is {}'.format(k))
print('Reconstruction error is {}'.format(diff))
    

