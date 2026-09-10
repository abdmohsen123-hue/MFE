

from src.MST import MST
from src.LeichtNewman import LeichtNewman
from src.helper import *
from src.MyUlamGalerkin import ulamGalerkin
import numpy as np
import heapq
from scipy.linalg import eig
import time
class SnapCluster:
    def __init__(self,Data,Disctrize_box_size,sim_time=np.inf,sample_time=1,U_scale=False,init_dt=0.05):
        t1=time.perf_counter()
        if U_scale:
            A,Box,P,mini,maxi,delta,dt_U_local_inv_dx=  ulamGalerkin(Data,Disctrize_box_size,sim=sim_time,sample=sample_time,ulam_scaled=U_scale,ulam_dt=init_dt)
            self.dt_U_local_inv_dx=dt_U_local_inv_dx
        A,Box,P,mini,maxi,delta=  ulamGalerkin(Data,Disctrize_box_size,sim=sim_time,sample=sample_time,ulam_scaled=U_scale,ulam_dt=init_dt)

        t2=time.perf_counter()
        print(f"Time for Ulam-Galerkin Disctrization:{t2-t1}")
        Boxlist=list(Box.values()) 
        print(f"Number of boxes: {len(Boxlist)}")
        CP=np.zeros(Data.shape[0],dtype=int) #cluster assignment for each point
        for i in range(len(Boxlist)):
            CP[Boxlist[i]]=i #Assign the box ID to each point in the trajectory
        CPT=[]
        CPT.append(CP)
        self.CPT=CPT
        self.Box=Box
        self.P=A
        self.mini=mini
        self.maxi=maxi
        self.BoxVolumeList=[(mini[0]+a*delta[0],mini[1]+b*delta[1]) for a,b in list(Box.keys())]
        self.delta=delta
        


    def compB(self,A): # compute the modularity matrix
        n,m = len(A),A.sum()
        #A   = (A!=0) + np.eye(n,dtype=int)
        Ki  = np.sum(A,0)               # in-degree
        Ko  = np.sum(A,1)               # out-degree
        b   = A - np.outer(Ko,Ki).T/m
        return b+b.T     # directed modularity matrix (symmetrized)
    def compQ(self,CPT,sim_time=np.inf,sample_time=1,U_scale=0): # compute the modularity and principal vector Q,S
        #print('length of incoming Bg = ',len(B))
        A = cluster_transition_matrix(CPT,sim=sim_time,sample=sample_time,dt_U_local_inv_dx=U_scale)
        B = self.compB(A)
        s   = np.ones(len(B),dtype=int)
        D,V = eig(B)
        v1  = V[:,np.argmax(np.real(D))]
        s   = np.sign(v1)
        return np.dot(s,np.dot(B,s))
    def cluster_counts(self,k=-1): #Compute the number of snapshots inside a cluster
        CP=self.CPT[k]
        nclust = max(CP) + 1
        cluster_counts = np.bincount(CP.astype(int), minlength=nclust)
        return cluster_counts
    def fixed_point_vector(self,ind=-1):
        cluster_counts = self.cluster_counts(k=ind)
        q=cluster_counts/sum(cluster_counts)
        return q
    def residence_time(self,CP):
      clusters=np.unique(CP)
      residence = [[] for _ in range(len(clusters))]
      count=1
      for i in range(len(CP)-1):
            if CP[i]==CP[i+1]:
                  count+=1
                  if CP[i]==CP[-1]:
                    residence[CP[i]].append(count)
            else:
                  residence[CP[i]].append(count)
                  count=1
      residence_time_res=[None]*len(clusters)
      for i,arr in enumerate(residence):
            residence_time_res[i]=np.mean(arr)
      return residence_time_res
    def save(self,Case):
        np.savetxt(
        os.path.join(Case, "CPT.csv"),
        self.CPT,
        delimiter=",",
        comments="",
        )

#Clustering algorithms
    def leicht_newman(self,data,fine_tune=False):
        return LeichtNewman(data,fine_tune)
    def leicht_newman_algorithm(self,fine_tune=False,sim_time=np.inf,sample_time=1,U_scale=0):
        t1=time.perf_counter()
        CP=self.CPT[-1]
        P=cluster_transition_matrix(self.CPT,sim=sim_time,sample=sample_time,dt_U_local_inv_dx=U_scale)
        F,c= LeichtNewman(P,fine_tune)
        C=list(c.values())
        CP2=np.zeros(len(CP),dtype=int)
        for i in range(len(C)):
            CP2[CP == i] = C[i]         #CP2 here has to give me point-cluster_2 ID 
        self.CPT.append(CP2)
        print(F"Number of clusters: {len(F)}")
        t2=time.perf_counter()
        print(f"Time for LeichtNewman:{t2-t1}")
        return self.CPT
    def repeated_leicht_newman_algorithm(self,Cluster_size=2,fine_tune=False,k=1,sim_time=np.inf,sample_time=1,U_scale=0):
        #Inputs: CP: cluster assignment
        #Outputs: CPT: list of cluster assignments for each iteration of reclustering
        t1=time.perf_counter()
        CP=self.CPT[-1]
        clustersize=np.zeros(max(CP))
        clustersize[0]=max(CP)+1
        it=0
        while it < clustersize[0] : 
            P=cluster_transition_matrix(self.CPT,sim=sim_time,sample=sample_time,dt_U_local_inv_dx=U_scale)
            F,c = LeichtNewman(P,fine_tune)
            C=list(c.values())    #C here tells cluster_1-Cluster_2 ID
            CP2=np.zeros(len(CP),dtype=int)
            for i in range(len(C)):
                CP2[CP == i] = C[i]         #CP2 here has to give me point-cluster_2 ID 
            it+=1
            clustersize[it]=len(F)
            if clustersize[it]==clustersize[it-1] or clustersize[it-1]<=Cluster_size:
                print(f"LeichtNewman Cluster assignments converged at iteration: {it}")
                print(F"Number of clusters: {len(F)}")
                t2=time.perf_counter()
                print(f"Time for LeichtNewman:{t2-t1}")
                return self.CPT
            self.CPT.append(CP2)
            CP=self.CPT[-1]
            print(f"LeichtNewman iteration: {it}")
            print(f"Number of old clusters: {len(c)}")
            print(f"Number of new clusters: {len(F)}")
    
    def mst(self,data,nclust, method='kruskal',cluster_type="Normal"):
        return MST(data,nclust,method,cluster_type)
    def mst_algorithm(self,nclust, method='kruskal',cluster_type="Normal",sim_time=np.inf,sample_time=1,U_scale=0):
        t1=time.perf_counter()
        CP=self.CPT[-1]
        P=cluster_transition_matrix(self.CPT,sim=sim_time,sample=sample_time,dt_U_local_inv_dx=U_scale)
        F,c,edges=MST(P,nclust,method,cluster_type)
        C=list(c.values())
        CP2=np.zeros(len(CP),dtype=int)
        for i in range(len(C)):
            CP2[np.where(CP==i)]=C[i]         
        self.CPT.append(CP2)
        t2=time.perf_counter()
        print(f"Time for MST:{t2-t1}")
        return self.CPT
    def repeated_mst_algorithm(self,Cluster_size,decrement=1,method="kruskal",cluster_type="Normal",sim_time=np.inf,sample_time=1,U_scale=0):
        #CPT.append(CPTinit)
        t1=time.perf_counter()
        CP=self.CPT[-1]
        Cluster_size_current=max(CP)+1
        while Cluster_size_current>Cluster_size:
            P=cluster_transition_matrix(self.CPT,sim=sim_time,sample=sample_time,dt_U_local_inv_dx=U_scale) 
            Cluster_size_current=Cluster_size_current-decrement #
            if Cluster_size_current%100==0:
                print(f"Current cluster size: {Cluster_size_current}")
            if Cluster_size_current<Cluster_size:
                Cluster_size_current=Cluster_size
            F,c,edges=MST(P,Cluster_size_current,method,cluster_type)
            C=list(c.values())
            CP2=np.zeros(len(CP),dtype=int)
            for i in range(len(C)):
                CP2[np.where(CP==i)]=C[i]         
            self.CPT.append(CP2)
            #print(np.array(CPT).shape)
            CP=self.CPT[-1]
        print(f"Number of clusters from MST: {max(CP)+1}")
        t2=time.perf_counter()
        print(f"Time for MST:{t2-t1}")
        return self.CPT
    def eigen_cluster(self,A,k=10):
        eigvals,vec = np.linalg.eig(A)
        # sort by real part descending
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[idx]
        vec = vec[:, idx]  # columns are eigenvectors, so sort columns
        #print(abs(vec))
        # vec shape should be (n_points, n_eigenvectors)
        clusters = [[] for _ in range(k)]
        for i in range(len(A)):
            #ind = np.argmax(np.abs(vec[i, :k]))  
            col_norms = np.linalg.norm(vec[:, :k], axis=0)
            projections = np.abs(vec[i, :k]) / col_norms
            ind = np.argmax(projections)
            clusters[ind].append(i)
        clusters = [c for c in clusters if len(c) > 0]

        c = dict(zip(list(range(k)),k*[0]))   #Vertices classification
        for i,cluster in enumerate(clusters):
            for ff in cluster: c[ff] = i
        return clusters,c
    def eigen_cluster_algorthm(self,nclust,sim_time=np.inf,sample_time=1,U_scale=0):
        t1=time.perf_counter()
        CP=self.CPT[-1]
        P=cluster_transition_matrix(CP,sim=sim_time,sample=sample_time,dt_U_local_inv_dx=U_scale)
        F,c=self.eigen_cluster(P,nclust)
        C=list(c.values())
        CP2=np.zeros(len(CP),dtype=int)
        for i in range(len(C)):
            CP2[np.where(CP==i)]=C[i]         
        self.CPT.append(CP2)
        t2=time.perf_counter()
        print(f"Time for eigen_cluster:{t2-t1}")
        return self.CPT

    def repeated_eigen_cluster_algorthm(self,Cluster_size,decrement=1,sim_time=np.inf,sample_time=1,U_scale=0):
        t1=time.perf_counter()
        #CPT.append(CPTinit)
        CP=self.CPT[-1]
        Cluster_size_current=max(CP)+1
        it=0
        while Cluster_size_current>Cluster_size:
            P=cluster_transition_matrix(CP,sim=sim_time,sample=sample_time,dt_U_local_inv_d=U_scale) 
            Cluster_size_current=Cluster_size_current-decrement #
            if it>300:
                print(f"Current cluster size: {Cluster_size_current}")
                it=0
            if Cluster_size_current<Cluster_size:
                Cluster_size_current=Cluster_size
            F,c=self.eigen_cluster(P,Cluster_size_current)
            C=list(c.values())
            CP2=np.zeros(len(CP),dtype=int)
            for i in range(len(C)):
                CP2[np.where(CP==i)]=C[i]         
            self.CPT.append(CP2)
            #print(np.array(CPT).shape)
            CP=self.CPT[-1]
            it=it+1
        t2=time.perf_counter()
        print(f"Time for eigen_cluster:{t2-t1}")
        return self.CPT