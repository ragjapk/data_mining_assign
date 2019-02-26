# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:12:30 2018

@author: IIST
"""
import numpy as np
import csv
from gradient_descent import find_rmse_gradient_descent
from gradient_descent import gradient_descent_ridge_reg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

rmsetrain=[0.00016575292767277483,0.00016606756430651592,0.0001667043095407814,0.00016800728806180314,0.0001707280037765341,0.00017660315597935652,0.0001898934924166777,0.00022126880559866858,0.00029563267404495797]
rmsevalidation=[0.00017128960702692054,0.00017160610203373613,0.00017224728366992183,0.00017356195241137118,0.00017631677688829337,0.00018229818213782168, 0.000195912369552796, 0.00022813058788819697, 0.00030421767020828727]
lambda_arr=[2**-16,2**-15,2**-14,2**-13,2**-12,2**-11,2**-10,2**-9,2**-8]
fig, ax = plt.subplots()
#ax2 = ax.twinx()
ax.plot(lambda_arr,rmsetrain,marker='o',label='train')
ax.plot(lambda_arr,rmsevalidation,marker='*',label='validation')
ax.legend(loc='best')
plt.show()