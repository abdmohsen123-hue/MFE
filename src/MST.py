# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import heapq
import networkx as nx
import random
from bisect import bisect_right


# %%
#MST Algorithm
def MST(A,nclust=1,method='kruskal',cluster_type="Normal"):
    AS=(A.T+A)  #Symmtrize the matrix
    #W=1-AS #Convert the similarity matrix to a weight matrix for the graph (the higher the similarity, the lower the weight)
    mask=np.where(AS>0)
    W=np.full_like(AS,np.inf)
    W[mask]=-np.log(AS[mask])
    #W =np.where(AS > 0, -np.log(AS + 1e-10), np.inf) #Convert the weight of the edges from being based on similarity to distinction (MST is based on disinction) 
    #print(2)
    n=len(W)           #number of vertices in the graph
    edges=[]           #list of edges in the graph (i,j,w) where i and j are the indices of the vertices and w is the weight of the edge between them
    for i in range(n): #record the weight of edges  into a list
        for j in range(n):
            if (j+i+1)>=n:
                break
            else:
                edges.append((i, j+i+1, W[i,j+i+1]))

    def kruskal(E,n):
        #Cycle detection using Union-Find data structure functions 
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]] 
                x = parent[x]
            return x
        def union(a,b):
            pa, pb = find(a), find(b)
            if pa == pb:
                return False
            parent[pa] = pb
            return True
        #Implementation of Kruskal's algorithm
        sorted_edges = sorted(E, key=lambda x: x[2])       #The original indices in increasing sorted order 
        mst_edges=[]
        for u,v,w in sorted_edges:
            if union(u,v):
                mst_edges.append((u,v,w))
            if len(mst_edges) == n-1:
                break
        return mst_edges
        
    def prims(E,n):
        dist = np.inf * np.ones(n)
        dist[0] = 0
        parent = -1 * np.ones(n, dtype=int)
        in_mst = np.zeros(n, dtype=bool)
        heap = [(dist[i],i) for i in range(n)]  # (weight, vertex)
        mst_edges = []
        while heap:
            w, u = heapq.heappop(heap)
            if in_mst[u]:
                continue
            in_mst[u] = True
            if parent[u] != -1:
                mst_edges.append((parent[u], u, w))
            for a, b, weight in E:
                v = None
                if a == u and not in_mst[b]:
                    v = b
                elif b == u and not in_mst[a]:
                    v = a
                if v is not None and weight < dist[v]:
                    dist[v] = weight
                    parent[v] = u
                    heapq.heappush(heap, (weight,v))
        mst_edges = []
        for i in range(1, n):
            mst_edges.append((parent[i], i, dist[i]))
        return mst_edges
    
    ######################################################################################################
    if method=='kruskal':
        mst_edges= kruskal(edges,n) #indices of edges in the MST
    elif method=='prims':
        mst_edges= prims(edges,n) #indices of edges in the MST
    ######################################################################################################

    #Clustering 
     #Clustering 
    if cluster_type=="boltzmann":
        #print("boltzmann clustering")
        for i in range(nclust-1):
            Z = sum(np.exp(w) for _,_,w in mst_edges)
            p_scale = [0.0]
            for _,_,w in mst_edges:
                p_scale.append(p_scale[-1] + np.exp(w)/Z)
            rand=random.random()
            idx = bisect_right(p_scale, rand) - 1
            mst_edges.pop(idx)                          #Remove the edge with the maximum weight from the MST
    else:
        for i in range(nclust-1):
            max_edge=max(mst_edges, key=lambda x: x[2]) #Find the index of the edge with the maximum weight in the MST
            mst_edges.remove(max_edge)                          #Remove the edge with the maximum weight from the MST
    
    graph = defaultdict(list) #Adjacency list representation of the graph
    for u,v,w in mst_edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    clusters = []
    for node in range(n):
        if node not in visited:
            stack = [node]
            comp = [] #list of vertices in the current cluster
            
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                
                visited.add(cur)
                comp.append(cur)
                
                for nb in graph[cur]:
                    if nb not in visited:
                        stack.append(nb)
            clusters.append(comp)
    c = dict(zip(list(range(n)),n*[0]))   #Vertices classification
    for i,cluster in enumerate(clusters):
        for ff in cluster: c[ff] = i
    pairs = [(i, j) for i, j, k in mst_edges]
    return clusters,c,pairs     #clusters: list of clusters, c: dictionary of vertices classification (0:cluster1, 1:cluster2,...),pairs: list of edges in the MST       



# %%
if __name__ == '__main__':
    Adj =np.array([
    [0,9,2,4,3,8,1,20,7,0],
    [9,0,0,0,0,1,0,6,0,0],
    [2,0,0,0,0,0,2,5,0,0],
    [4,0,0,0,0,0,3,0,0,0],
    [3,0,0,0,0,5,0,0,25,4],
    [8,1,0,0,5,0,0,0,0,0],
    [1,0,2,3,0,0,0,0,0,0],
    [20,6,5,0,0,0,0,0,0,0],
    [7,0,0,0,25,0,0,0,0,7],
    [0,0,0,0,4,0,0,0,7,0]
    ])
    A,c,edge_colors=MST(Adj,3,'prims') # Prim's algorithm
    # Create a graph
    G = nx.from_numpy_array(Adj)

    # Position the nodes
    pos = {
        0: (0,0),
        1: (1,1),
        2: (1,-1),
        3: (-1,-1),
        4: (-1,1),
        5: (0,1),
        6: (0,-1),
        7: (1,0),
        8: (-1,0),
        9: (-2,1)
    }
    # Draw edge labels (weights)
    highlight_edges = edge_colors

    edge_colors = [
        'red' if (u,v) in highlight_edges or (v,u) in highlight_edges else 'gray'
        for u, v in G.edges()
    ]
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightgreen', edge_color=edge_colors)

    # Extract edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')


    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_color='red')
    #nx.draw_planar(G,with_labels=True, node_size=1000, node_color='lightgreen')
    # Show plot
    plt.show()

# %%
if __name__ == '__main__':

    nC = 10           # number of clusters
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
    #A=csr_matrix(A)
    B = A[P, :][:, P]                  # shuffling the matrix 

    plt.figure(2)    # scrambled adjacency matrix 
    plt.ion()
    plt.spy(B)
    plt.show()
    c = np.zeros(len(B), dtype=int)
    w,c,ind=MST(B,nC,"prims") # Prim's algorithm
    #print(w[9])
    #F,c = LeichtNewman_sparse(B) # Leicht-Newman algorithm
    Q = np.zeros((nT,nT),dtype=int)
    for i,j in enumerate(sum(w,[])): Q[i,j] = 1
    
    C = Q@(B@Q.T)    # clustering matrix (re-organization) 

    plt.figure(3)   # reclustered adjacency matrix 
    plt.ion()
    plt.spy(C)
    plt.show()
    n = 20
    P = np.zeros((n,n))

    clusters = [
        list(range(0,7)),
        list(range(7,14)),
        list(range(14,20))
    ]

    # Fill strong connections inside clusters
    for cluster in clusters:
        for i in cluster:
            for j in cluster:
                if i != j:
                    P[i,j] = 0.12

    # Weak connections between clusters
    for i in range(n):
        for j in range(n):
            if P[i,j] == 0 and i != j:
                P[i,j] = 0.005

    # Normalize rows to make it a Markov matrix
    P = P / P.sum(axis=1, keepdims=True)



# %%


# %%



