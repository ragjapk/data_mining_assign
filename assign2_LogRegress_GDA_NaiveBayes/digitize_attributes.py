# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:05:14 2018

@author: IIST
"""

import numpy as np
import csv
from sklearn.model_selection import train_test_split
from untitled1 import find_performance_measures_gda
import math
def getdata(): 
    reader = csv.reader(open("adultdata.csv", "rt"), delimiter=",")
    x = list(reader)
    result = np.array(x).astype("float")
    unique_attributes(result)
    
def unique_attributes(data):
    a,c=np.unique(data,return_counts=True)
    return(a)
        
getdata()