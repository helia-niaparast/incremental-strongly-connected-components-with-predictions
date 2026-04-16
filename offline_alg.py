"""
Offline Incremental Strongly Connected Components

Based on: https://github.com/kth-competitive-programming/kactl/blob/main/content/graph/SCC.h
and https://codeforces.com/blog/entry/91608

Usage:
    adj = [[] for _ in range(n)]
    # Build adjacency list
    num_sccs, scc_id = sccs(adj)
    
    # For offline incremental SCC
    edges = [[u, v], [a, b], ...]  # List of edges
    joins = offline_incremental_scc(edges, n)

scc_id[v] = id, where 0 <= id < num_sccs
For each edge u -> v: scc_id[u] >= scc_id[v]

Time: O(n + m) for SCC, O((n + m) log m) for offline incremental
Space: O(n) for SCC, O(n + m) for offline incremental
"""

from typing import List, Tuple

def sccs(adj: List[List[int]]) -> Tuple[int, List[int]]:
    """
    Find strongly connected components using Tarjan's algorithm.
    
    Args:
        adj: Adjacency list representation of the graph
        
    Returns:
        Tuple of (number of SCCs, SCC ID for each vertex)
    """
    n = len(adj)
    num_sccs = 0
    q = 0  # timestamp counter
    s = 0  # stack pointer
    scc_id = [-1] * n
    tin = [0] * n  # discovery time
    st = [0] * n   # stack
    
    def iterative_dfs(v: int) -> int:
        nonlocal num_sccs, q, s

        # work arrays for the iterative simulation of recursion
        low_work = [0] * len(adj)             # running low-link value per node
        stack = []                            # frames: [node, next_child_index]

        # "enter" v
        tin[v] = q + 1
        q += 1
        st[s] = v
        s += 1
        low_work[v] = tin[v]
        stack.append([v, 0])

        ret_low = None  # will hold the return value for the root call

        while stack:
            node, i = stack[-1]  # peek

            if i < len(adj[node]):
                u = adj[node][i]
                stack[-1][1] = i + 1  # advance child index

                if scc_id[u] < 0:
                    if tin[u] == 0:
                        # "recurse" into u: initialize its frame
                        tin[u] = q + 1
                        q += 1
                        st[s] = u
                        s += 1
                        low_work[u] = tin[u]
                        stack.append([u, 0])
                    else:
                        # back/forward/cross edge to a discovered but unassigned node
                        low_work[node] = low_work[node] if low_work[node] < tin[u] else tin[u]
                # else: u already assigned to an SCC; ignore (matches your original)
                continue

            # finished all neighbors of `node`: finalize like in the recursive epilogue
            low = low_work[node]
            if tin[node] == low:
                while scc_id[node] < 0:
                    s -= 1
                    scc_id[st[s]] = num_sccs
                num_sccs += 1

            # pop frame
            stack.pop()

            if stack:
                parent = stack[-1][0]
                # propagate child's low up to parent
                if low_work[parent] > low:
                    low_work[parent] = low
            else:
                # node was the root of this DFS call
                ret_low = low

        return ret_low


    for i in range(n):
        if tin[i] == 0:
            iterative_dfs(i)
    
    return num_sccs, scc_id


def offline_incremental_scc(edges: List[List[int]], n: int) -> List[int]:
    """
    Offline incremental SCC algorithm.
    
    Args:
        edges: List of edges, where each edge is [u, v]
        n: Number of vertices
        
    Returns:
        joins[i] = minimum prefix of edges [0, joins[i]] for
        edges[i][0] and edges[i][1] to be in the same SCC
        joins[i] = m if they're never in the same SCC
        joins[i] = -1 if edges[i][0] == edges[i][1]
    """
    m = len(edges)
    ids = [-1] * n
    joins = [m] * m
    idx = list(range(m))
    vs = [0] * n
    scc_id = []
    adj = []
    
    def divide_and_conquer(el: int, er: int, tl: int, tr: int):
        nonlocal adj, scc_id
        
        adj.clear()
        mid = (tl + tr) // 2
        
        # Build adjacency list for current range
        for it in range(el, er):
            edge_idx = idx[it]
            u, v = edges[edge_idx][0], edges[edge_idx][1]
            
            # Map vertices to compressed indices
            for w in [u, v]:
                if ids[w] == -1:
                    ids[w] = len(adj)
                    vs[len(adj)] = w
                    adj.append([])
            
            u_mapped, v_mapped = ids[u], ids[v]
            edges[edge_idx][0], edges[edge_idx][1] = u_mapped, v_mapped
            
            # Add edge if it's in the left half
            if edge_idx <= mid:
                adj[u_mapped].append(v_mapped)
        
        # Reset vertex mapping
        for i in range(len(adj)):
            ids[vs[i]] = -1
        
        # Find SCCs
        _, scc_id = sccs(adj)
        
        # Partition edges based on whether endpoints are in same SCC
        split_idx = el
        for it in range(el, er):
            edge_idx = idx[it]
            u, v = edges[edge_idx][0], edges[edge_idx][1]
            if scc_id[u] == scc_id[v]:
                # Move to left partition
                idx[it], idx[split_idx] = idx[split_idx], idx[it]
                split_idx += 1
        
        # Set join time for edges in same SCC
        for it in range(el, split_idx):
            joins[idx[it]] = mid
        
        if tr - tl == 1:
            return
        
        # Update edge endpoints to SCC IDs for recursive calls
        for it in range(split_idx, er):
            edge_idx = idx[it]
            u, v = edges[edge_idx][0], edges[edge_idx][1]
            edges[edge_idx][0], edges[edge_idx][1] = scc_id[u], scc_id[v]
        
        # Recursive calls
        divide_and_conquer(el, split_idx, tl, mid)
        divide_and_conquer(split_idx, er, mid, tr)
    
    # Handle self-edges
    for i in range(m):
        if edges[i][0] == edges[i][1]:
            joins[i] = -1
    
    # Use -1 as lower bound to correctly handle self-edges
    divide_and_conquer(0, m, -1, m)
    
    return joins