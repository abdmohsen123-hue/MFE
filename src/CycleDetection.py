from collections import defaultdict
import numpy as np

def find_cycles_backtracking(A,limit=np.inf,cap=0):
    M = A.copy()
    n = len(A)
    cycles = []
    mask=np.where(M<cap)
    M[mask]=0
    
    np.fill_diagonal(M, 0)
    
    #M = M / M.sum(axis=0,keepdims=True)
    def dfs(start, current, path):
        path.append(current)
        if len(path) < limit: # Limit the depth of the search to avoid infinite loops

            for nxt in range(n):

                if M[nxt,current] == 0:
                    continue

                # Found a cycle
                if nxt == start and len(path) > 1:
                    cycles.append(path[:] + [start])
                    

                # Continue searching
                elif nxt not in path:
                    dfs(start, nxt, path)

        path.pop()

    
    for start in range(n):
        dfs(start, start, [])
    

    # Remove duplicate cycles
    unique = []
    seen = set()

    for cycle in cycles:

        c = cycle[:-1]           # remove repeated start

        # rotate so smallest vertex comes first
        m = min(c)
        k = c.index(m)
        c = c[k:] + c[:k]

        t = tuple(c)

        if t not in seen:
            seen.add(t)
            unique.append(c + [c[0]])
            
    cycles_probablity =[]
    for cycle in unique:
        prob=1.0
        for ind in range(len(cycle)-1):
            prob*=A[cycle[ind+1],cycle[ind]]
        cycles_probablity.append((cycle, prob))
    cycles_probablity.sort(key=lambda x: x[1], reverse=True)

    return cycles_probablity






def johnsons_algorithm(A, cap=0):
    mask=np.where(A<cap)
    M = A.copy()
    M[mask]=0
    np.fill_diagonal(M, 0)
    #M = M / M.sum(axis=0,keepdims=True)
    

    graph={}
    for i in range(M.shape[0]):
        graph[i]=list(np.where(M[:,i]>0)[0])
    """Find all elementary cycles in a directed graph."""
    def strong_connect(v,subgraph, index_counter, stack, lowlinks, index, on_stack, sccs):
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in subgraph.get(v, []):
            if w not in index:
                strong_connect(w, subgraph, index_counter, stack, lowlinks, index, on_stack, sccs)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack.get(w, False):
                lowlinks[v] = min(lowlinks[v], index[w])

        if lowlinks[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1 or (len(scc) == 1 and v in graph.get(v, [])):
                sccs.append(scc)

    def find_sccs(subgraph, vertices):
        sccs = []
        index = {}
        lowlinks = {}
        on_stack = {}
        stack = []
        index_counter = [0]
        for v in vertices:
            if v not in index:
                strong_connect(v,subgraph, index_counter, stack, lowlinks, index, on_stack, sccs)
        return sccs

    def circuit(v, s, adj, blocked, B, stack, cycles):
        f = False
        stack.append(v)
        blocked[v] = True

        for w in adj.get(v, []):
            if w == s:
                cycles.append(list(stack[:] + [s]))
                f = True
            elif not blocked.get(w, False):
                if circuit(w, s, adj, blocked, B, stack, cycles):
                    f = True

        if f:
            unblock(v, blocked, B)
        else:
            for w in adj.get(v, []):
                if v not in B[w]:
                    B[w].add(v)

        stack.pop()
        return f

    def unblock(u, blocked, B):
        blocked[u] = False
        for w in list(B[u]):
            if blocked.get(w, False):
                unblock(w, blocked, B)
        B[u] = set()

    vertices = sorted(graph.keys())
    cycles = []

    for s in vertices:
        subgraph_vertices = [v for v in vertices if v >= s]
        sub_adj = {v: [w for w in graph.get(v, []) if w >= s]
                   for v in subgraph_vertices}

        sccs = find_sccs(sub_adj, subgraph_vertices)
        scc_with_s = None
        for scc in sccs:
            if s in scc:
                scc_with_s = set(scc)
                break

        if scc_with_s:
            adj = {v: [w for w in sub_adj.get(v, []) if w in scc_with_s]
                   for v in scc_with_s}
            blocked = {}
            B = defaultdict(set)
            stack = []
            for v in scc_with_s:
                blocked[v] = False
            circuit(s, s, adj, blocked, B, stack, cycles)
    cycles_probablity =[]
    for cycle in cycles:
        prob=1.0
        for ind in range(len(cycle)-1):
            prob*=A[cycle[ind+1],cycle[ind]]
        cycles_probablity.append((cycle, prob))
    cycles_probablity.sort(key=lambda x: x[1], reverse=True)
    return cycles_probablity




from concurrent.futures import ProcessPoolExecutor


def _dfs_worker(args):
    start, M, limit = args
    n = len(M)
    cycles = []

    def dfs(current, path):
        path.append(current)

        if len(path) < limit:
            for nxt in range(n):

                if M[nxt, current] == 0:
                    continue

                if nxt == start and len(path) > 1:
                    cycles.append(path[:] + [start])

                elif nxt not in path:
                    dfs(nxt, path)

        path.pop()

    dfs(start, [])
    return cycles


def find_cycles_backtracking_parallel(A, limit=np.inf, cap=0, workers=None):
    M = A.copy()
    M[M < cap] = 0
    np.fill_diagonal(M, 0)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            _dfs_worker,
            [(start, M, limit) for start in range(len(M))]
        )

    cycles = []
    for r in results:
        cycles.extend(r)

    # Remove duplicate cycles
    unique = []
    seen = set()

    for cycle in cycles:
        c = cycle[:-1]

        m = min(c)
        k = c.index(m)
        c = c[k:] + c[:k]

        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique.append(c + [c[0]])

    cycles_probability = []
    for cycle in unique:
        prob = 1.0
        for i in range(len(cycle) - 1):
            prob *= A[cycle[i + 1], cycle[i]]
        cycles_probability.append((cycle, prob))

    cycles_probability.sort(key=lambda x: x[1], reverse=True)

    return cycles_probability