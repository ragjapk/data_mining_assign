# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:17:22 2019

@author: IIST
"""

from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

df = pd.read_csv('data.csv',header=None)
df.replace('?', np.nan, inplace= True)
df = df.fillna(method='ffill')
data2=df.values
dataset = np.array(data2, dtype='float')
row,col=dataset.shape
X=np.delete(dataset,col-1,axis=1)
clustering=DBSCAN( eps = 500, metric="euclidean", min_samples = 4, n_jobs = -1).fit(X)
indices=np.argwhere(clustering.labels_==-1)
print(len(indices))
data3=np.delete(dataset, indices,axis=0)
np.savetxt("adaboost_dataset.csv", data3, delimiter=",")
