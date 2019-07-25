# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:33:48 2019

@author: IIST
"""
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
reader = csv.reader(open("slice_localization_pca_regression.csv", "rt"), delimiter=",")
x = list(reader)
regress = np.array(x).astype("float")
row,col=regress.shape
result=np.delete(regress,col-1,axis=1)
y=np.take(regress, col-1, axis=1)

reg = LinearRegression()
reg.fit(result, y.ravel())
y_pred=reg.predict(result)
print('score for original datas score {}'.format(reg.score(result, y)))


