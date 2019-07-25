# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:40:48 2019

@author: IIST
"""
import numpy as np
def forward(N,A,B,pi,T,O):
    alpha=np.zeros((T,N))
    #initialization:
    alpha[0]=np.multiply(pi[0],B[:,O[0]])    
    #induction:
    for t in range(1,T):
       for j in range(N):
           sum=0
           for i in range(N):
               sum=sum+(alpha[t-1,i]*A[i,j]*B[j,O[t]])
           alpha[t,j]=sum
             
    #termination:
    return alpha

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
           beta[t,i]=sum
             
    #termination:
    sum=0
    for i in range(N):
        sum=sum+(pi[i]*B[i,O[0]]*beta[0,i])
    return beta

def initialize():
    N=4
    T=22
    M=2
    A = np.random.rand(N,N)
    A=A/A.sum(axis=1)[:,None]
    B = np.random.rand(N,M)
    B=B/B.sum(axis=1)[:,None]
    pi=np.random.rand(N,1) 
    pi=pi/pi.sum()
    O=[0,0,1,1,1,0,0,0,0,0,0,1,0,1,0,1,0,1,1,1,1,1]
    return M,N,A,B,pi,T,O

M,N,A,B,pi,T,O=initialize()
max_iter=0
gamma=np.zeros((T,N))
eta=np.zeros((T,N,N))
prob=np.inf
norm=20
iteration=0
Anew=np.zeros((N,N))
while(iteration<50):
    iteration=iteration+1
    #E-Step:
    alpha=forward(N,A,B,pi,T,O)
    beta=backward(N,A,B,pi,T,O)
    for t in range(T-1):
        for i in range(N):
            gamma[t,i]=alpha[t,i]*beta[t,i]/(np.sum(alpha[t]*beta[t]))
            for j in range(N):
                eta[t,i,j]=alpha[t,i]*A[i,j]*B[j,O[t+1]]*beta[t+1,j]
        eta[t,:,:]=eta[t,:,:]/np.sum(eta[t,:,:])
    #M-Step:
    pi=gamma[1,:]
    for i in range(N):
        for j in range(N):
            Anew[i,j]=np.sum(eta[:,i,j])/np.sum(gamma[:,i])
        for k in range(M):
            summ=0
            for t in range(T-1):
                if(O[t]==k):
                    summ=summ+gamma[t,i]
            B[i,k]=summ/np.sum(gamma[:,i])
    norm=np.linalg.norm(Anew-A)
    A=np.copy(Anew)
    new_prob=forward(N,A,B,pi,T,O)   
    #print(np.sum(new_prob[T-1]))
    newprob=np.sum(new_prob[T-1])
    if(prob<newprob):
        print('New probability is:', np.sum(new_prob[T-1]))
        #break
    prob=np.sum(new_prob[T-1])
    max_iter=max_iter+1
    
#print(np.sum(new_prob[T-1]))
print(A)
print(B)
print(pi)