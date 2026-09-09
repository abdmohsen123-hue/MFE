import heapq
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
from src.LeichtNewman import *
from scipy.optimize import minimize

import os
import networkx as nx
import matplotlib.colors as mcolors

def create_case_folder(dt,N,Re,Lx,Lz):

    folder_name = f"Case_dt{dt}_N{N}_Re{Re}_Lx{Lx:.2f}_Lz{Lz:.2f}"
    
    # Check if folder exists
    if not os.path.exists(f"result/{folder_name}"):
        os.makedirs(f"result/{folder_name}")
        return f"result/{folder_name}"
    
    # If exists, find the next available number
    return f"result/{folder_name}"

def cluster_transition_matrix(CPT,k=-1,sim=np.inf,sample=1,dt_U_local_inv_dx=None):  #Make the adjacency matrix for reclustering based on the cluster assignments of the previous iteration
    CP=CPT[k]
    if np.isfinite(sim):
        boundary = set(np.arange(sim, len(CP), sample))
    else:
        boundary = set()
    #Inputs: CP: cluster assignment
    #Outputs: Pij: Markov Transition Matrix  
    M=np.zeros((max(CP)+1,max(CP)+1))     #start the markov matrix
    for i in range(len(CP)-1):
        if i+1 in boundary:
            continue
        M[CP[i+1],CP[i]]=M[CP[i+1],CP[i]]+1 #transition from j to i 
    if dt_U_local_inv_dx is not None:
        idx = np.diag_indices_from(M)
        M[idx] *= dt_U_local_inv_dx
        M=np.floor(M)
    M = M / M.sum(axis=0,keepdims=True)
    return M
    #M[0,0]=1                                        #first snapshot assignment
    #M[CP[-1],CP[-1]]=1 #Assigment of the last snapshot since there is no transition from the last snapshot to any other snapshot, we assign it to itself
    #if CP[i]!=CP[i+1]:                            #if there is a transition from cluster j to i
        #M[CP[i+1],CP[i]]=M[CP[i+1],CP[i]]+1 #transition from j to i
def CTM_power(CTM, it):
    CTM_power = np.linalg.matrix_power(CTM, it)
    return CTM_power

def cluster_colors(CPT,k=-1,type=1):
    CP=CPT[k]
    nclust = len(np.unique(CP))
    if type==1:
        c  = cm.tab20(np.linspace(0, 1, nclust))  # color coding
    else:
        c  = cm.tab10(np.linspace(0, 1, nclust))  # color coding
    colors = [c[label % len(c)] for label in CP]
    return colors,c
def cluster_centroids(CPT,Data,k=-1):
    CP=CPT[k]
    nclust = len(np.unique(CP))
    CC=np.zeros((nclust,Data.shape[1]))
    for i in range(nclust):
        mask = (CP == i)
        CC[i] = Data[mask].mean(axis=0) if np.sum(mask) > 0 else np.zeros(Data.shape[1])
    return CC


def reduceP(P,F):
    Z = compZ(F)
    P = Z.T@P@Z
    print('reduced to ',len(P))
    for i,p in enumerate(P): P[i,:] = p/sum(p) 
    return P 

def compZ(F): 
    nC = len(F)
    nT = max(sum(F,[]))+1
    Z  = np.zeros((nT,nC))
    for i in range(nC):
        f = F[i]
        for j in f: Z[j,i] = 1
    return Z 
def extreme_points(DE, threshold=1):
    mu=np.mean(DE,axis=0)
    sigma=np.std(DE,axis=0)
    #print(f"mean:{mu},std:{sigma}")
    critical_points = mu + threshold * sigma
    extreme = np.zeros(DE.shape, dtype=bool)
    for i in range(DE.shape[1]):
        extreme[:, i] = (
            np.abs(DE[:, i] - mu[i])
            > threshold * sigma[i]
        )
    extreme_idx = np.where(np.all(extreme, axis=1))[0]
    return np.array(extreme_idx,dtype=int),critical_points
def extreme_clusters(CPT, DE, threshold=1,k=-1):
    CP=CPT[k]
    ex_points,critical_points= extreme_points(DE,threshold)
    return np.unique(CP[ex_points])
def box_cluster_transition_matrix(CPT,sim_time=np.inf,sample_time=1,k=-1,U_scale=0):
        CP=CPT[k]
        CP0=CPT[0]
        bx_clusters = {}
        for i in range(len(CP)):
                if CP0[i] not in bx_clusters:
                        bx_clusters[CP0[i]] = CP[i]
        sorted_bx_clusters = dict(sorted(bx_clusters.items(), key=lambda item: item[1]))
        sorted_indices=list(sorted_bx_clusters.keys())

        P0M=cluster_transition_matrix(CPT,k=0,sim=sim_time,sample=sample_time,dt_U_local_inv_dx=U_scale)
        P0M=P0M[np.ix_(sorted_indices, sorted_indices)]
        return P0M,bx_clusters
def compute_Modularity(CPT,sim_time=np.inf,sample_time=1,U_scale_2=0):
    A=cluster_transition_matrix(CPT,k=0,sim=sim_time,sample=sample_time,dt_U_local_inv_dx=U_scale_2)
    #np.fill_diagonal(A,0)
    P0M,bx_clusters=box_cluster_transition_matrix(CPT,sim_time,sample_time,U_scale=U_scale_2)
    bx_clusters_list=list(bx_clusters.keys())
    Ki  = np.sum(A,0)               # in-degree
    Ko  = np.sum(A,1)               # out-degree
    m = A.sum()                 # total weight of edges
    Q = 0.0
    for i in bx_clusters_list:
        for j in bx_clusters_list:
            if bx_clusters[i] == bx_clusters[j]:  # Check if nodes i and j are in the same cluster
                Q += A[i,j] - (Ko[i] * Ki[j]) / m
    Q /= m
    return Q
def solve_a0(Dstar, Estar, Lx=1.75*np.pi, Lz=1.2*np.pi, method='L-BFGS-B'):
    a_init = np.random.randn(9) * 0.1
    a,b,c = 2*np.pi/Lx, np.pi/2, 2*np.pi/Lz
    pi4 = np.pi*np.pi/4
    D1  = pi4
    D2  = pi4 + c*c
    D3  = np.pi*np.pi + 2*c*c
    D4  = a*a + pi4
    D5  = a*a + pi4
    D6  = 2*(a*a + pi4 + c*c)
    D7  = 2*(a*a + pi4 + c*c)
    D8  = 3*(a*a + pi4 + c*c)
    D9  = 9*pi4
    Lambda = np.array([D1, D2, D3, D4, D5, D6, D7, D8, D9])
    
    a_init = np.asarray(a_init, dtype=float)

    def objective(a):
        E = np.sum(a**2)
        D = np.sum(Lambda * a**2)
        rD = D - Dstar
        rE = E - Estar
        f = rD**2 + rE**2
        return f

    def gradient(a):
        E = np.sum(a**2)
        D = np.sum(Lambda * a**2)
        rD = D - Dstar
        rE = E - Estar
        grad = 2.0 * a * (rD * Lambda + rE)
        return grad

    res = minimize(objective, a_init, jac=gradient, method=method)
    return res.x, res