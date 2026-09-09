from src.helper import *

def dijkstra_shortest_path(CPT,source,target,cluster=-1,sim=np.inf,sample=1,U_scale=0):
    A=cluster_transition_matrix(CPT,sim=sim,sample=sample,k=cluster,dt_U_local_inv_dx=U_scale)
    W = np.full_like(A, np.inf, dtype=float)
    mask = A > 0
    W[mask] = -np.log(A[mask])
    n=len(W)           #number of vertices in the graph
    edges=[]           #list of edges in the graph (i,j,w) where i and j are the indices of the vertices and w is the weight of the edge between them
    for i in range(n): 
        for j in range(n):
            edges.append((i,j, W[i,j]))
    #edges=[x for x in edges if x[2]>=0] #edges=(i:sink,j:source,w:weight) where w>0()
    #print("change")
    def dijkstra(E,n,source,target):
        dist = np.inf * np.ones(n)
        dist[source] = 0
        parent = -1 * np.ones(n, dtype=int)
        in_mst = np.zeros(n, dtype=bool)
        heap = [(dist[i],i) for i in range(n)]  # (weight, vertex)
        heapq.heapify(heap)
        while heap:
            w, u = heapq.heappop(heap)          # (weight, vertex)
            if in_mst[u] or w > dist[u]:
                continue
            in_mst[u] = True
            if u == target:
                break
            for i, j, weight in E:   # j >> i is the direction of the edge
                v = None
                if j == u and not in_mst[i]:
                    v = i
                if v is not None and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    parent[v] = u   
                    heapq.heappush(heap, (dist[v],v))
        path = []
        current = target
        while current != -1:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path, np.exp(-dist[target])
    ######################################################################################################
    return dijkstra(edges,n,source,target)
    ######################################################################################################

def dijkstra_algorithems(CPT, start=0,sim_time=np.inf,sample_time=1,clusters=-1,U_scale=0):
    A=cluster_transition_matrix(CPT,k=clusters,sim=sim_time,sample=sample_time,dt_U_local_inv_dx=U_scale)
    W = np.full_like(A, np.inf, dtype=float)
    mask = A > 0
    W[mask] = -np.log(A[mask])
    n=len(W)           #number of vertices in the graph
    edges=[]           #list of edges in the graph (i,j,w) where i and j are the indices of the vertices and w is the weight of the edge between them
    for i in range(n): 
        for j in range(n):
            if ((j+1)*(i+1))>=n*n:
                break
            else:
                edges.append((i, j, W[i,j]))
    #edges=[x for x in edges if x[2]>0] #edges=(i:sink,j:source,w:weight) where w>0
    def dijkstra(E,n,start):
        dist = np.inf * np.ones(n)
        dist[start] = 0
        parent = -1 * np.ones(n, dtype=int)
        in_mst = np.zeros(n, dtype=bool)
        heap = [(dist[i],i) for i in range(n)]  # (weight, vertex)
        heapq.heapify(heap)
        mst_edges = []
        while heap:
            w, u = heapq.heappop(heap)          # (weight, vertex)
            if in_mst[u] or w > dist[u]:
                continue
            in_mst[u] = True
            if parent[u] != -1:
                mst_edges.append((parent[u], u, w))
            for i, j, weight in E:   # j >> i is the direction of the edge
                v = None
                if j == u and not in_mst[i]:
                    v = i
                if v is not None and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    parent[v] = u   
                    heapq.heappush(heap, (dist[v],v))
        spt_edges = []
        for i in range(n):
            spt_edges.append((i, parent[i], dist[i]))
        return spt_edges
    ######################################################################################################
    return dijkstra(edges,n,start) 
    ######################################################################################################
