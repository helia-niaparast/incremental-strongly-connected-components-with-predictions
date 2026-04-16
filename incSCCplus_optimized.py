#the optimized version of IncSCC+ algorithm in "Incremental graph computations: Doable and undoable." by Fan, Hu, Tian 2017
#tarjan's imp @mmcilree's github

from collections import defaultdict, deque
from typing import List, Tuple, Set

def tarjan_scc(adj: List[List[int]]) -> Tuple[int, List[int]]:
    """
    runs Tarjan's, returns number of SCCs and list mapping each node to its SCC ID.
    """
    n = len(adj)
    num_sccs = 0
    q = 0  # timestamp counter
    stack = []
    on_stack = [False] * n
    scc_id = [-1] * n
    tin = [0] * n  # discovery time
    low = [0] * n  # lowlink values
        
    def dfs_iterative(start: int):
        nonlocal num_sccs, q
        # Call stack for DFS frames: each item is (v, next_child_index)
        call_stack = [(start, 0)]
        if tin[start] == 0:
            q += 1
            tin[start] = low[start] = q
            stack.append(start)
            on_stack[start] = True

        while call_stack:
            v, i = call_stack[-1]  # peek

            # If we still have neighbors to process
            if i < len(adj[v]):
                u = adj[v][i]
                # Advance v's next_child_index for the next time we see this frame
                call_stack[-1] = (v, i + 1)

                if tin[u] == 0:
                    # Tree edge: "recurse" into u by pushing a new frame.
                    q += 1
                    tin[u] = low[u] = q
                    stack.append(u)
                    on_stack[u] = True
                    call_stack.append((u, 0))
                elif on_stack[u]:
                    # Back/forward/cross edge that hits the active stack
                    low[v] = min(low[v], tin[u])
                # else (visited but not on_stack): it's already assigned to some SCC; ignore
                continue

            # No more neighbors of v → finish v ("post-visit" work)
            call_stack.pop()

            # If v is a root of an SCC, pop the Tarjan stack
            if low[v] == tin[v]:
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc_id[w] = num_sccs
                    if w == v:
                        break
                num_sccs += 1

            # Propagate low-link to parent (the frame now at the top, if any).
            if call_stack:
                parent, _ = call_stack[-1]
                low[parent] = min(low[parent], low[v])

    for i in range(n):
        if tin[i] == 0:
            dfs_iterative(i)
    
    return num_sccs, scc_id

class IncrementalSCC:
    def __init__(self, adj: List[List[int]]):
        """
        initialize with original graph adjacency list.
        build Tarjan structures and contracted graph.
        """
        self.adj = adj
        self.n = len(adj)
        
        # initial SCC computation
        self.num_sccs, self.scc_id = tarjan_scc(adj)
        self.scc_nodes = defaultdict(set) #a mapping from each scc_id to its nodes
        for v,id in enumerate(self.scc_id):
            self.scc_nodes[id].add(v)
        # build contracted graph Gc and ranks
        self.build_contracted_graph()
        self.assign_ranks()
    
    def build_contracted_graph(self):
        """
        construct contracted graph Gc where each SCC is a node.
        Also updates self.num_sccs to the number of unique SCCs.
        """
        self.Gc = defaultdict(set)
        unique_sccs = set(self.scc_nodes.keys())
        self.num_sccs = len(unique_sccs)
        for v in range(self.n):
            for w in self.adj[v]:
                sv, sw = self.scc_id[v], self.scc_id[w]
                if sv != sw:
                    self.Gc[sv].add(sw)
        # also maintain reverse edges for DFSb
        self.Gc_rev = defaultdict(set)
        for u, neighbors in self.Gc.items():
            for v in neighbors:
                self.Gc_rev[v].add(u)

    def assign_ranks(self):
        """
        topological sort of Gc to assign ranks -- a node has higher rank if it must come later.
        Ensures all SCC IDs in self.scc_id are present in self.rank.
        """
        #unique_sccs = set(self.scc_id)
        unique_sccs = set(self.scc_nodes.keys())
        self.num_sccs = len(unique_sccs)
        indegree = {u: len(self.Gc_rev[u]) for u in unique_sccs}
        q = deque([u for u, d in indegree.items() if d == 0])
        self.rank = {}
        r = 0
        while q:
            u = q.popleft()
            self.rank[u] = r
            r += 1
            for v in self.Gc[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.append(v)
        # check all SCC IDs in self.scc_id are present in self.rank
        for scc in unique_sccs:
            if scc not in self.rank:
                self.rank[scc] = r
                r += 1
    
    def dfs_forward(self, start: int, cutoff: int) -> Set[int]:
        """
        DFS in Gc forward from start, stopping at nodes with rank > cutoff.
        returns set of reachable scc IDs.
        """
        visited = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in visited or self.rank[u] > cutoff:
                continue
            visited.add(u)
            for v in self.Gc[u]:
                stack.append(v)
        return visited
    
    def dfs_backward(self, start: int, cutoff: int) -> Set[int]:
        """
        DFS in Gc backward from start, stopping at nodes with rank < cutoff.
        returns set of reachable scc IDs in reverse graph.
        """
        visited = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in visited or self.rank[u] < cutoff:
                continue
            visited.add(u)
            for v in self.Gc_rev[u]:
                stack.append(v)
        return visited
    
    def merge_sccs(self, components: Set[int]):
        """
        merge a set of SCC nodes into one in contracted graph.
        updates scc_id, Gc, Gc_rev, and ranks.
        """
        new_id = min(components)
        for old in components:
            if old == new_id:
                continue
            # update scc_id in original mapping
            """ for i in range(self.n):
                if self.scc_id[i] == old:
                    self.scc_id[i] = new_id """
            for v in self.scc_nodes[old]:
                self.scc_id[v] = new_id
            self.scc_nodes[new_id].update(self.scc_nodes[old])
            del self.scc_nodes[old]
            # update edges in Gc
            for v in self.Gc[old]:
                if v != old:
                    self.Gc_rev[v].discard(old)
                    self.Gc_rev[v].add(new_id)
            for v in self.Gc_rev[old]:
                if v != old:
                    self.Gc[v].discard(old)
                    self.Gc[v].add(new_id)    
            self.Gc[new_id].update(self.Gc[old])
            self.Gc[new_id].discard(old)
            self.Gc_rev[new_id].update(self.Gc_rev[old])
            self.Gc_rev[new_id].discard(old)
            del self.Gc[old]
            del self.Gc_rev[old]
        self.Gc[new_id].discard(new_id)
        self.Gc_rev[new_id].discard(new_id)
    
    def realloc_ranks(self, affl: Set[int], affr: Set[int]):
        """
        reallocate ranks for nodes in the affected area (affl U affr)
        by swapping their positions in the topological order.
        """
        affected = affl | affr #the union of two sets
        intersection = affl & affr #the nodes in the new SCC (if any)
        # get current ranks and sort affl and affr by their current rank
        affl_pure = affl - intersection
        affr_pure = affr - intersection
        affl_pure_sorted = sorted(affl_pure, key=lambda x: self.rank[x])
        affr_pure_sorted = sorted(affr_pure, key=lambda x: self.rank[x])
        # get all ranks in the affected region, sorted
        all_ranks = sorted([self.rank[x] for x in affected])
        # assign lowest ranks to affl, highest to affr, preserving order
        for i, node in enumerate(affl_pure_sorted):
            self.rank[node] = all_ranks[i]
        for i, node in enumerate(affr_pure_sorted):
            self.rank[node] = all_ranks[i + len(affl)]
        if len(intersection) > 0:
            new_id = min(intersection)
            self.rank[new_id] = all_ranks[1 + len(affl_pure)]    
    
    def add_new_edge(self, v: int, w: int):
        """
        implement IncSCC+ for insertion of edge (v, w).
        v, w are node indices in original graph.
        """
        sv, sw = self.scc_id[v], self.scc_id[w]
        
        # case 1: Same-SCC
        if sv == sw:
            return
        
        # add edge to contracted graph (compressed SCC graph)
        self.Gc[sv].add(sw)
        self.Gc_rev[sw].add(sv)
        # add edge to original graph as well
        self.adj[v].append(w)
        
        # case 2: No-cycle insertion
        if self.rank[sv] < self.rank[sw]:
            return
        
        # case 3: Potential-cycle case
        # find forward and backward affected sets
        affr = self.dfs_forward(sw, self.rank[sv])
        affl = self.dfs_backward(sv, self.rank[sw])
        
        #find node of the new SCC
        intersection = affr & affl #if a cycle has formed, these are the nodes in the new SCC
        if len(intersection) > 0: #there is a cycle
            self.merge_sccs(intersection)
        #reallocate ranks
        self.realloc_ranks(affl, affr)

        #clean up the absorbed nodes
        if len(intersection) > 0:
            new_id = min(intersection)
            for node in intersection:
                if node != new_id:
                    del self.rank[node]