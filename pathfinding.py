import heapq
import math
import time

# Dataframe: x, y, elevation, index (aka row number) -- each row is unique
# Dictionary:
#   Input: give index
#   Output: indices of the neighbor(s)

# start: x, y coordinate
# end: x, y coordinate

# OUTPUT: set of paths with COSTS and edges/nodes

# HELPER FUNCTIONS   ------ START -------

# Heuristic functions
def mahattan_distance(p1, p2) -> int:
    x,y,_ = p1
    gx,gy,_ = p2
    return (abs(gx-x)+abs(gy-y))

def euclidean_distance(p1, p2):
    x,y,_ = p1
    gx,gy,_ = p2
    return math.sqrt((gx-x)**2 + (gy-y)**2)

def slope(p1, p2) -> int:
    x,y,z = p1
    gx,gy,gz = p2
    dist = euclidean_distance(p1, p2)
    if (dist == 0):
        return 0
    return math.arctan((gz-z)/dist)

def elevation(p1, p2) -> int:
    x,y,z = p1
    gx,gy,gz = p2
    return abs(gz-z)

# Find neighbors
def return_neighbors(neighbors, idx):
	return neighbors[idx]

# Usage
with open('neighbors.pickle', 'rb') as f:
    neighbors = pickle.load(f)
ns = return_neighbors(neighbors, idx)
print(ns)

# Find index given a x, y coordinate
def return_index(nodes, x, y):
	return nodes[nodes['x'] == x, nodes['y'] == y].index.value
# E.g.
return_index(nodes, '34.0987426', '-117.7030676')

def path_selection(OPEN_List, nodeCosts):
    # select thing from open that is not dominated
    # delete from open
    # move gn from G_op to G_close
    pass

# Helper functions ------ END -------

# Multiobjective A* search
# Input: start -- x,y
# goal -- x,y
def astar(dataframe, start, goal):

    # 1. Create
    # Initialize list of alternatives triple(n, g , F(n,g))
    OPEN_List = {start: (start, (0,0), (euclidean_distance(start, goal), elevation(start, goal)))} # (s, gs, H(s,gs))
    # Initialize two empty sets
    GOALN = []
    COSTS = [] # potentially dictionary of nodes with costs

    # Dictionary -- mapping nodes to Gop(n) && Gcl(n)
    nodeCosts = {
        start: ([(0,0)],[], None)
    }
    # 2. Check Termination
    # if OPEN is empty, then backtrack in SG from the nodes in GOALN
    #   && return the set of solution paths w/ costs in COSTS
    if not OPEN_List:
        return backtrack(GOALN)

    # 3. Path Selection
    while OPEN_List:
        # Select an alternative (n, gn, F) from OPEN_List w/ f in F nondominated in OPEN_List
        node, (n_dist, n_ele), (n_dist_heu, n_ele_heu), nodeCosts = path_selection(OPEN_List, nodeCosts)
        # Delete (n, gn, F) from OPEN, and move gn from Gop(n) go Gcl(n)

        # 4. Solution Recording
        # If n is goal, then
        if n == goal:
            # Include n in GOALN and gn in COSTS
            GOALN += [n]
            COSTS += [(n_dist, n_ele)]
            # Eliminate from OPEN_List all alternatives (x, gx, Fx) such that all vectors in Fx
            # such that all vectors in Fx are dominated by gn (FILTERING)
            OPEN_List = prune_list(OPEN_List)
            # Go back to step 2 (removed)


        # 5. Path Expansion
        # If n is not goal, then
        else:
            # For all successors nodes m of n that do not produce cycles in SG do:
            for neighbor in return_neighbors(node):
                # (a) Calculate the cost of the new path found to m: gm = gn + c(n,m)
                next_dist, next_ele = n_dist + euclidean_distance(node, neighbor), n_ele + elevation(node, neighbor)



                # (b) If m is a new node
                if neighbor not in OPEN_List:
                    # i. Calculate Fm = F (m, gm) filtering estimates dominated by COSTS
                    F_dist, F_ele = next_dist + euclidean_distance(neighbor, goal), next_ele + elevation(neighbor, goal)
                    # ii. If Fm is not empty, put (m, gm, Fm) in OPEN,
                    #  and put gm in Gop(m) labelling a pointer to n
                    OPEN_List[neighbor] = (neighbor, (next_dist, next_ele), (F_dist, F_ele))
                    nodeCosts[neighbor] = ([(next_dist, next_ele)],[], node)
                    # iii. Go to step 2
                # else (m is not a new node), in case
                else:
                    # - gm is in Gop(m) or gm is in Gcl(m):
                    if (next_dist, next_ele) in nodeCosts[neighbor][0] or (next_dist, next_ele) in nodeCosts[neighbor][1]:

                        # label with gm a pointer to n && go to step 2
                        nodeCosts[neighbor][2] += node
        #           - If gm is non-dominated by any cost vectors in Gop(m) U Gcl(m)
        #               (a path to m with new cost had been found), then:
        #               i. Eliminate from Gop(m) and Gcl(m) vectors dominated by gm
        #               ii. Calculate Fm = F(m, gm) filtering estimates dominated by COSTS
        #               iii. If Fm is not empty, put (m, gm, Fm) in OPEN_List, and put gm in Gop(m)
        #                   labelling a pointer to n.
        #               iv. Go to step 2.
        #            Otherwise: go to step 2
