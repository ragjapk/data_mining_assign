# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:10:26 2019

@author: IIST
"""

import numpy as np
def exhaustive(N,A,B,pi,T,O):
    
def question2():
    N=3
    A=np.array([[.2,.2,.6],
   [.3,.3,.4],
   [.2,.5,.3]])

    B=np.array([[0.7,0.3],
   [0.5,0.5],
   [0.8,0.2]])

    pi=[0.3,0.4,0.3]
    T=2
    O=[1,1]
    return N,A,B,pi,T,O

def question3():
    N=2
    A=np.array([[.5,.5],[.3,.7]])
    B=np.array([[0.6,0.2,0.2],
                [0.2,0.5,0.3]])
    pi=[0.8,0.2]
    T=2
    O=[0,1]
    return N,A,B,pi,T,O

ans=question3()    
exhaustive(ans[0],ans[1],ans[2],ans[3],ans[4],ans[5])