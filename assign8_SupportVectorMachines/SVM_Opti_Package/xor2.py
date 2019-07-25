# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:30:10 2019

@author: IIST
"""

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn.svm import SVC
X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
y = np.array([-1,1,1,-1])
h = 0.02  # step size in the mesh

X_test_p = X[y == 1]
X_test_n = X[y == -1]

# we create an instance of SVM and fit our data. We do not scale our
# data since we want to plot the support vectors

clf = SVC(gamma='auto')
clf.fit(X, y)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
fig, ax = plt.subplots()
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
ax.axis('off')

# Plot also the training points
plt.scatter(X_test_p[:,0],X_test_p[:,1],c='blue')
plt.scatter(X_test_n[:,0],X_test_n[:,1],c='red')

ax.set_title('XOR')