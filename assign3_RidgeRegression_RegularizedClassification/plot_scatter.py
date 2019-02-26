# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:39:20 2018

@author: IIST
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 21:36:32 2018

@author: IIST
"""

from sympy import symbols
from sympy import plot
from mpl_toolkits import mplot3d
from sympy.plotting import plot3d
from numpy import linspace
from sympy import lambdify
import matplotlib.pyplot as mpl

import matplotlib.pyplot as plt
import csv
import numpy as np

from sympy import Symbol

reader = csv.reader(open("unmanned.csv", "rt"), delimiter=",")
data = list(reader)
result = np.array(data).astype("float")
t_rows,t_cols=result.shape
#Normalization
for i in range(0,t_cols):
    mean=np.mean(result[:,i])
    std_dev=np.std(result[:,i])
    print(mean)
    print(std_dev)
    for j in range(0,t_rows):
        result[j,i]=(result[j,i]-mean)/std_dev
ys=result[:,1]
xs=result[:,0]
zs=result[:,2]
fig = plt.figure()
ax = plt.axes(projection='3d')

for i in range(1,len(ys)):
    ax.scatter(xs[i],ys[i],zs[i])
#p1._backend.save('im_line.png')
#p1._backend.save('im.png')