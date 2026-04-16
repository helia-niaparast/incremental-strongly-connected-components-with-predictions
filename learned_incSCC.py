#official implementation of the Learned IncSCC algorithm in the paper "Incremental Strongly Connected Components with Predictions"

from typing import List, Tuple

def find_sccs(adj: List[List[int]]) -> Tuple[int, List[int]]:
    #runs Tarjan's, returns number of SCCs and list mapping each node to its SCC ID.
    
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

class Union_Find:
    #the Node Label Data Structure described in the paper
    def __init__(self, n):
        self.roots = set(range(n)) #each root is the canonical node of its SCC. Initially, each node is the root of its own SCC.
        self.canonicals = list(range(n)) #canonicals[i] is the canonical node (aka root) of node i
        self.num_sccs = n
        self.sizes = {i:1 for i in range(n)} #a map from each root to the size of its SCC
        self.children = {i:[] for i in range(n)} #a map from each root to the set of its children, i.e., the set of nodes in its SCC

    #merges the SCCs containing v and u
    def merge(self, v, u):
        v_root, u_root = self.canonicals[v], self.canonicals[u]
        if v_root == u_root:
            return
        v_size, u_size = self.sizes[v_root], self.sizes[u_root]
        if v_size > u_size:
            u, v = v, u
            u_root, v_root = v_root, u_root
            u_size, v_size = v_size, u_size

        #v is always in the smaller component
        self.children[v_root].append(v_root) #just to avoid repeating the steps for v_root
        for c in self.children[v_root]:
            self.canonicals[c] = u_root
            self.children[u_root].append(c)
        self.sizes[u_root] += self.sizes[v_root]

        self.num_sccs -= 1
        del self.sizes[v_root]
        del self.children[v_root]
        self.roots.remove(v_root)

    def __str__(self):
        ret = "Canonical Nodes: "
        for v, root in enumerate(self.canonicals):
            ret += "(" + str(v) + " : " + str(root) + "), "
        return ret

class Subproblem:
    def __init__(self, l, r, parent, predicted_arrivals):
        self.n = 0 #number of nodes; the nodes are numbered from 0 to n-1
        self.edges: List[List[int]] = [] #mapped edges. The endpoints are in {0,...,n-1}
        self.original_edges: List[List[int]] = [] #original edges according to the original node labels
        self.smallest_nodes = [] #list of size n that determines the original smallest label in each of the n (super) nodes
        self.updated_n = 0 #number of nodes after the clean-up. We always have updated_n <= n
        self.scc_id = [] #list of size updated_n
        self.num_sccs = 0
        self.updated_smallest_nodes = [] #list of size num_sccs that determines the original smallest label in each of the SCCs
        self.node_map = [] #the map used in clean-up from {0,...,n-1} to {0,...,updated_n-1}
        self.reverse_map = [] #reverse of the node_map
        self.predicted_arrivals = predicted_arrivals

        self.l = l #inclusive
        self.r = r #exlusive
        self.mid = (self.r+self.l)//2
        self.parent = parent
        self.right_child = None
        self.left_child = None

    
    def build(self):
        #computes the updated info (right and left edges and nodes) for its right and left children
        
        if self.r - self.l == 1:
            return

        #clean up: deleting the nodes that are not used
        adj = []
        self.node_map = [-1] * self.n #map from {0,...n-1} to {0,...,len(adj)-1}
        self.reverse_map = [-1] * self.n
        for [u,v],[original_u,original_v] in zip(self.edges,self.original_edges):
            for w in [u,v]:
                if self.node_map[w] == -1:
                    self.node_map[w] = len(adj)
                    self.reverse_map[len(adj)] = w
                    adj.append([])
            u_mapped, v_mapped = self.node_map[u], self.node_map[v]
            if self.predicted_arrivals[(original_u, original_v)] <= self.mid:
                adj[u_mapped].append(v_mapped)

        self.updated_n = len(adj) #number of nodes after the clean-up
        self.num_sccs, self.scc_id = find_sccs(adj) #make sure the scc_id's start from 0
        self.updated_smallest_nodes = [-1] * self.num_sccs

        #update the smallest node in each SCC
        for v in range(self.updated_n):
            v_unmapped = self.reverse_map[v]
            if self.updated_smallest_nodes[self.scc_id[v]] == -1 or (self.updated_smallest_nodes[self.scc_id[v]] != -1 and self.updated_smallest_nodes[self.scc_id[v]] > self.smallest_nodes[v_unmapped]):
                self.updated_smallest_nodes[self.scc_id[v]] = self.smallest_nodes[v_unmapped]

        right_edges = []
        right_original_edges = []
        left_edges = []
        left_original_edges = []

        for [u,v],[original_u,original_v] in zip(self.edges,self.original_edges):
            u_mapped, v_mapped = self.node_map[u], self.node_map[v]
            if self.scc_id[u_mapped] == self.scc_id[v_mapped]: #this edge goes to left child
                if self.predicted_arrivals[(original_u, original_v)] <= self.mid:
                    left_edges.append([u_mapped,v_mapped])
                    left_original_edges.append([original_u, original_v])
            else: #this edge goes to right child
                right_edges.append([self.scc_id[u_mapped],self.scc_id[v_mapped]])
                right_original_edges.append([original_u, original_v])

        if self.left_child == None:
            self.left_child = Subproblem(self.l, self.mid, self, self.predicted_arrivals)
        if self.right_child == None:
            self.right_child = Subproblem(self.mid, self.r, self, self.predicted_arrivals)

        self.left_child.n = self.updated_n
        self.left_child.edges = left_edges
        self.left_child.original_edges = left_original_edges
        self.left_child.smallest_nodes = [self.smallest_nodes[self.reverse_map[v]] for v in range(self.updated_n)]

        self.right_child.n = self.num_sccs
        self.right_child.edges = right_edges
        self.right_child.original_edges = right_original_edges
        self.right_child.smallest_nodes = self.updated_smallest_nodes

class Online_Problem:
    def __init__(self, n, predictions, actual_edges):
        self.n = n #number of nodes. nodes are labeled with 0 to n-1
        self.predictions = predictions #the predicted sequence of edges in the form of list of [u,v]'s
        self.actual_edges = actual_edges #the actual input sequence
        self.root = None #the root of the recursive tree of subproblems
        self.current_subproblem = None
        self.initial_predicted_arrivals = {} #same as predicted arrivals but does not get updated
        self.predicted_arrivals = {} #maps predicted edges to timestamps starting from time 1. Assumes there are no duplicate edges.

        for i,[u,v] in enumerate(self.predictions):
            self.predicted_arrivals[(u,v)] = i+1
        self.union_find = Union_Find(n)

    def run_algorithm(self):
        #build the root
        m = len(self.predictions)
        root = Subproblem(0, m+1, None, self.predicted_arrivals)
        self.root = root
        self.current_subproblem = root
        self.initialize_root()

        for i,[u,v] in enumerate(self.actual_edges):
            real_time = i+1
            predicted_time = self.predicted_arrivals.get((u,v))
            if predicted_time == None:
                m = len(self.predictions)
                predicted_time = m + 1
            self.update_predictions(real_time, u, v)
            lca = self.LCA(predicted_time) #this includes both cases of correct and incorrect predictions. Also handles the first edge.
            if lca == self.root:
                self.initialize_root()
            self.rebuild_path(lca, real_time)
        return

    def initialize_root(self):
        self.root.n = self.n
        self.root.edges = self.predictions
        self.root.original_edges = self.predictions
        self.root.smallest_nodes = list(range(self.n))

    def LCA(self, t_hat):
        #finds LCA of current subproblem and subproblem t_hat
        current_node = self.current_subproblem
        while current_node != self.root and t_hat >= current_node.r:
            current_node = current_node.parent
        return current_node

    def update_predictions(self, t, u, v):
        predicted_time = self.predicted_arrivals.get((u,v))
        m = len(self.predictions)
        if predicted_time == None:
            predicted_time = m + 1

        if predicted_time != t:
            if predicted_time == m+1:
                a,b = self.predictions[m-1]
                del self.predicted_arrivals[(a,b)]
                predicted_time -= 1
            for j in range(predicted_time-1, t-1, -1):
                self.predictions[j] = self.predictions[j-1]
                w,z = self.predictions[j]
                self.predicted_arrivals[(w,z)] = j+1
            self.predictions[t-1] = [u,v]
            self.predicted_arrivals[(u,v)] = t

    def query(self, v):
        return self.union_find.canonicals[v]

    def rebuild_path(self, subproblem, t):
        #rebuild the path from subproblem to G_t
        subproblem.build()
        if t == subproblem.mid:
            #update the global SCC array based on the new merges
            num_sccs = subproblem.num_sccs
            sccs = [[] for _ in range(num_sccs)]
            for v in range(subproblem.updated_n):
                sccs[subproblem.scc_id[v]].append(v)
            for l in sccs:
                if len(l) > 1:
                    v = l[0]
                    for u in l[1:]:
                        v_unmapped = subproblem.reverse_map[v]
                        u_unmapped = subproblem.reverse_map[u]
                        self.union_find.merge(subproblem.smallest_nodes[v_unmapped], subproblem.smallest_nodes[u_unmapped])
            self.current_subproblem = subproblem
            return
        elif t > subproblem.mid:
            self.rebuild_path(subproblem.right_child, t)
        else:
            self.rebuild_path(subproblem.left_child, t)
        return