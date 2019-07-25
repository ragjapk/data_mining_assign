# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:28:05 2019

@author: IIST
"""

from sklearn.preprocessing import scale 
import numpy as np
import smo_copy as smo
#import chunking
#import SMO
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

X = np.genfromtxt('data4.csv', delimiter=',')

output = X[:,-1]
input = X[:,0 : X.shape[1] - 1]
input_norm = scale(input)

kernel = "Polynomial"         
w, b, train_index, test_index, alpha_vec, _ = smo.compute(input_norm,output,kernel)

train_inp = input[train_index]


#normal = w
plt.subplot(2, 2, 1)
posclass =input[output>0]
negclass =input[output<0]
plt.subplots_adjust(wspace=0.4, hspace=0.4)

h = 0.5

x_min=input[:, 0].min() - 1
x_max =input[:, 0].max() + 1

y_min = input[:, 1].min() - 1
y_max = input[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

#z = (-normal[0] * xx - normal[1] * yy + b) * 1. /normal[2]
#fig = plt.figure()
#plt3d = fig.gca(projection='3d')
#plt3d.plot_surface(xx, yy, z)

test_set = np.c_[xx.ravel(), yy.ravel()]
predicted = np.zeros(np.shape(test_set)[0])

predicted = smo.get_predicted(test_set,input[train_index],alpha_vec,output[train_index],b)


predicted = predicted.reshape(xx.shape)      
plt.contourf(xx, yy, predicted, cmap=plt.cm.terrain, alpha=0.5,linewidths=0)  
plt.scatter(posclass[:, 0], posclass[:, 1],c='red')
plt.scatter(negclass[:, 0], negclass[:, 1],c='blue')
for i in np.where(alpha_vec != 0)[0]:
    plt.scatter(train_inp[i][0],train_inp[i][1],c='black',marker ="^",s=30)

plt.show()

#chunking.compute(input,output,kernel)
#SMO.compute(input,output,kernel)
            


        
    
    

