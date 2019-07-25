# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 12:08:45 2019

@author: IIST
"""

import numpy as np

def backward(N,A,B,pi,T,O):
    beta=np.zeros((T,N))
    #initialization:
    for i in range(N):
        beta[T-1,i]=1
    #induction:
    for t in range(T-2,-1,-1):
        for i in range(N):
           sum=0
           for j in range(N):
               sum=sum+(beta[t+1,j]*A[i,j]*B[j,O[t+1]])
               print(beta[t+1,j],A[i,j],B[j,O[t+1]])
           beta[t,i]=sum
             
    #termination:
    sum=0
    for i in range(N):
        print(pi[i],B[i,O[0]],beta[0,i])
        sum=sum+(pi[i]*B[i,O[0]]*beta[0,i])
    print(sum)
    print(beta)

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
    T=4
    O=[0,0,1,2]
    return N,A,B,pi,T,O
def question4():
    N=2
    A=np.array([[.7,.3],[.2,.8]])
    B=np.array([[0.1,0.2,0,0.5,0.2],
                [0.3,0.2,0.4,0,0.1]])
    pi=[0.5,0.5]
    T=10
    O=[2,0,1,2,1,3,4,4,1,2]
    return N,A,B,pi,T,O
ans=question3()    
backward(ans[0],ans[1],ans[2],ans[3],ans[4],ans[5])