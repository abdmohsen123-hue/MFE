#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from scipy.sparse.linalg import eigsh

# In[11]:


#Dense Matrix
def compB(A): # compute the modularity matrix
    n,m = len(A),A.sum()
    #A   = (A!=0) + np.eye(n,dtype=int)
    Ki  = np.sum(A,0)               # in-degree
    Ko  = np.sum(A,1)               # out-degree
    b   = A - np.outer(Ko,Ki).T/m
    return b+b.T     # directed modularity matrix (symmetrized)

def compQ(B): # compute the modularity and principal vector Q,S
    #print('length of incoming Bg = ',len(B))
    s   = np.ones(len(B),dtype=int)
    D,V = eig(B)
    v1  = V[:,np.argmax(np.real(D))]
    s   = np.sign(v1)
    return np.dot(s,np.dot(B,s)),s
def splitCommunity(B,ind,finetune=0):  # split the sub-community (indices ind)
    Bp  = B[ind,:][:,ind]   # extract the sub-graph
    Bg  = Bp - np.diag(np.sum(Bp,1)) # subtract row sum from diag
    #Bg  = Bg + Bg.T         # symmetrize
    q,s = compQ(Bg)         # compute modularity, principal vector
    if (q>1e-10):
        if finetune==1:
            s=fine_tune(Bg,s)
        ind1=[i for j,i in enumerate(ind) if s[j]==1] # split ind return ind that are positive
        ind2=[i for j,i in enumerate(ind) if s[j]==-1] # return ind that are negative
        return ind1,ind2
    return [],[]
def fine_tune(B, s):
    st = s.copy()
    while True:
        current_q = st @ B @ st
        best_delta = 0.0
        best = -1
        for i in range(len(st)):
            s_try = st.copy()
            s_try[i] *= -1
            q_try = s_try @ B @ s_try
            delta = q_try - current_q

            if delta > best_delta:
                best_delta = delta
                best = i

        if best == -1:  # no improving flip
            break
        st[best] *= -1
    return st

def LeichtNewman(A, finetune=0):           # Leicht-Newman algorithm A adjacent Matrix No fine tuning
    if finetune==1:
        print("Fine tuning is on")
    B,n = compB(A),len(A)      # compute modularity matrix B
    W,F = [list(range(n))],[]  # initialize cluster work and final lists
    while W:
        w = W.pop(0)                      # work on first entry in W take first community
        i1,i2 = splitCommunity(B,w,finetune)       # split the indices i1 cluster 1, i2 cluster 2
        if (len(i1)!=0) & (len(i2)!=0):   # legitimate split
            W = [i1]+[i2]+W  
        else:                             # impossible split 
            F.append(w)                   # transfer w to final list F
    c = dict(zip(list(range(n)),n*[0]))   #Vertices classification
    for i,f in enumerate(F):
        for ff in f: c[ff] = i
    return F,c #F is a list of comminiuties[[V1,V3],[V2,V4],[V6,V6],...], c is a dictionary of vertices ID V2 belong to F[1] 
def quality(A,c):
    n = len(A)
    m = A.sum()  # total edges (directed)
    Ki  = np.sum(A,0)               # in-degree
    Ko  = np.sum(A,1)               # out-degree
    Q = 0
    for i in range(n):
        for j in range(n):
            if c[i] == c[j]:
                Q += A[i,j] - (Ki[i]*Ko[j])/(m)
    Q /= (m)
    return Q



if __name__ == '__main__':

    nC = 200           # number of clusters
    nS = [np.random.randint(20,80,dtype=int) for i in range(nC)] # cluster sizes
    nT = sum(nS)     # total size 
    p  = np.random.permutation(nT)   # permutation 

    A,i = np.zeros((nT,nT),dtype=int),0
    for n in nS:
        X = (np.random.random((n,n))<0.66).astype(int)
        A[i:i+n,i:i+n] = X      # block-diagonal matrix 
        i += n

    plt.figure(1)    # original block-diagonal adjacency matrix 
    plt.ion()
    plt.spy(A)
    plt.show()
    P = np.random.permutation(A.shape[0])  # 1D array  random permutation matrix 
    A=csr_matrix(A)
    B = A[P, :][:, P]                  # shuffling the matrix 

    plt.figure(2)    # scrambled adjacency matrix 
    plt.ion()
    plt.spy(B.toarray())
    plt.show()
    c = np.zeros(len(B.toarray()), dtype=int)

    F,c = LeichtNewman_sparse(B) # Leicht-Newman algorithm
    Q = np.zeros((nT,nT),dtype=int)
    for i,j in enumerate(sum(F,[])): Q[i,j] = 1

    C = Q@(B@Q.T)    # clustering matrix (re-organization) 

    plt.figure(3)   # reclustered adjacency matrix 
    plt.ion()
    plt.spy(C)
    plt.show()




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




