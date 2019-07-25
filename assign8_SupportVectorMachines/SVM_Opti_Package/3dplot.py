# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:53:47 2019

@author: IIST
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
import csv
rs = np.random.RandomState(1234)

# Generate some fake data.
#n_samples = 4
# X is the input features by row.
reader = csv.reader(open("Data3_new.csv", "rt"), delimiter=",")
x = list(reader)
H = np.array(x).astype("float")
X=np.delete(H,-1,axis=1)
Y=np.take(H, -1, axis=1)

X_test_p = X[Y == 1]
X_test_n = X[Y == -1]
# Fit the data with an svm
svc = SVC(kernel='linear')
svc.fit(X,Y)
reader = csv.reader(open("f_data3new.csv", "rt"), delimiter=",")
x = list(reader)
f = np.array(x).astype("float")
b=-12.26
# The equation of the separating plane is given by all x in R^3 such that:
# np.dot(svc.coef_[0], x) + b = 0. We should solve for the last coordinate
# to plot the plane in terms of x and y.


h = 0.5

x_min=X[:, 0].min()- 1
x_max =X[:, 0].max() + 1

y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
z = (-f[0] * xx - f[1] * yy + b) * 1. /f[2]
# Plot stuff.
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, z)
ax.scatter(X_test_p[:,0],X_test_p[:,1],X_test_p[:,2],c='blue')
ax.scatter(X_test_n[:,0],X_test_n[:,1],X_test_p[:,2],c='red')
plt.show()