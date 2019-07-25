# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 06:49:47 2019

@author: IIST
"""

import numpy as np
from scipy import stats 

rms1= np.genfromtxt('rmse_iterative.csv')
rms2= np.genfromtxt('rmse_online.csv')
#rms2= np.genfromtxt('rmse_ridge.csv')
print(len(rms1))

m_rmse1= np.mean(rms1)
m_rmse2= np.mean(rms2)

var1=np.var(rms1,ddof=1)
var2=np.var(rms2,ddof=1)

m=len(rms1)
n=len(rms2)

var_1=var1/m
var_2=var2/n

t=(m_rmse1-m_rmse2)/np.sqrt(var_1+var_2)

num=(var_1+var_2)**2
den1=(var_1**2)/(m-1)
den2=(var_2**2)/(n-1)
r=int(num/(den1+den2))

print(stats.ttest_ind(rms1,rms2,equal_var = False))