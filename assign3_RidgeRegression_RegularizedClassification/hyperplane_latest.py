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
from sympy import plot

from sympy import Symbol
reader = csv.reader(open("w_iteration_method.csv", "rt"), delimiter=",")
x = list(reader)
w = np.array(x).astype("float")

reader = csv.reader(open("Data2.csv", "rt"), delimiter=",")
data = list(reader)
result = np.array(data).astype("float")
t_rows,t_cols=result.shape
#Normalization
for i in range(0,t_cols-1):
    mean=np.mean(result[:,i])
    std_dev=np.std(result[:,i])
    print(mean)
    print(std_dev)
    for j in range(0,t_rows):
        result[j,i]=(result[j,i]-mean)/std_dev
ys=result[:,1]
xs=result[:,0]
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.scatter(x,y,z)
#ax.plot3D(x,y,z)

x2=int(w[1])
x1=int(w[0])
x, y = symbols('x y')
eq = x2*y + x1
#eq=x**3 - 3*x*y**2
print(len(ys))
p1=plot(eq)
for i in range(1,len(ys)):
    p1._backend.ax.scatter(xs[i],ys[i])
p1._backend.save('im_line.png')
#p1._backend.save('im.png')