"""
Created on Sat Feb 23 13:23:58 2019

@author: IIST
"""
import numpy as np

def vieterbi(N,A,B,pi,T,O):
    delta=np.zeros((T,N))
    psi=np.zeros((T,N))
    final_seq=np.zeros(T,dtype='uint8')
    #Initialization
    delta[0]=pi*B[:,O[0]]
    #for i in range(N):
        #print(pi[i],B[i,O[0]])
    #psi[0]
    #Induction    
    for t in range(1,T):
       for j in range(N):
           sum=0
           max_val=np.NINF
           for i in range(N):
               val=delta[t-1,i]*A[i,j]*B[j,O[t]]
               #print(delta[t-1,i],A[i,j],B[j,O[t]])
               if(val>max_val):
                   max_val=val
                   arg=i
           delta[t,j]=max_val
           psi[t,j]=arg
    #Termination
    max_prob=np.max(delta[T-1])
    final_seq[T-1]=np.argmax(delta[T-1])
    #print(final_seq[T-1])
    for t in range(T-2,-1,-1):    
        final_seq[t]=psi[t+1,final_seq[t+1]]
    print(final_seq)
    #print(psi)
    #print(delta)
def question2():
    N=3
    A=np.array([[.2,.2,.6],
   [.3,.3,.4],
   [.2,.5,.3]])

    B=np.array([[0.7,0.3],
   [0.5,0.5],
   [0.8,0.2]])

    pi=[0.3,0.4,0.3]
    T=3
    O=[0,0,1]
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
vieterbi(ans[0],ans[1],ans[2],ans[3],ans[4],ans[5])