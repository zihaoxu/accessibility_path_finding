import math, time, pickle
import numpy as np
import pandas as pd

####### Turn list into string and use as key in dict!!!!!!

# Galbal paths
dataset_path = '0-datasets/'
raw_path = dataset_path + 'raw/'
mst_path = dataset_path + 'mst/'

# Unit Conversions
lat_to_m = 111 * 1000
lng_to_m = 111.3 * 1000

# Load in necessary files
nodes_df = pd.read_csv(mst_path + 'clean_node.csv')
nodes_df.index = range(len(nodes_df))
with open(mst_path + 'neighbors.pickle', 'rb') as f:
    neighbors = pickle.load(f)

# ------ START: HELPER FUNCTIONS -------
# Heuristic function 1: Euclidean distance between two points
def euclidean_distance(p1, p2):
    x,y,z = p1
    gx,gy,gz = p2
    y_diff_m = abs(gy - y) * lat_to_m
    x_diff_m = abs(gx - x) * lng_to_m
    return math.sqrt(y_diff_m**2 + x_diff_m**2 + abs(gz-z)**2)

# Heuristic function 2: Elevation difference between two points
def elevation(p1, p2) -> int:
    x,y,z = p1
    gx,gy,gz = p2
    return abs(gz-z)

# Slope between two points
def slope(p1, p2) -> int:
    x,y,z = p1
    gx,gy,gz = p2
    y_diff_m = abs(gy - y) * lat_to_m
    x_diff_m = abs(gx - x) * lng_to_m
    dist = math.sqrt(y_diff_m**2 + x_diff_m**2)
    if (dist == 0):
        return 0
    return np.arctan((gz-z)/dist)/math.pi * 180

# Find neighbors
# Input: neighbors list, index of a point
# Output: list of indices (of neighbors)
def return_neighbors(neighbors, idx):
	return neighbors[idx]

# Find coordinates of node with corresponding index:
# Input: nodes dataframe, index
# Output: x,y,z coords of node with that index
def get_node_coords(nodes_df, index):
    return float(nodes_df.loc[index, 'x']), \
           float(nodes_df.loc[index, 'y']), \
           float(nodes_df.loc[index, 'elevation'])

# Check if heuristic2 dominates heuristic1
# Input: two sets of heuristic values
# Output: Boolean indicating whether 2 dominates 1
def check_dominance(heuristic1, heuristic2):
	h11, h12 = heuristic1
	h21, h22 = heuristic2
	return h21 < h11 and h22 < h12

# Selects a path to explore next -- pick a nondominated value
# Input: open_list of values
#        G_open, G_close (Gop & Gcl from write up)
# Output: a item from open_list, updated open_list, updated G_open, G_close
def path_selection(open_list, G_open, G_close):
    if len(open_list) == 1:
        node_info = open_list.pop(0)
        (n, (n_dist, n_ele), (n_dist_heu, n_ele_heu)) = node_info
        G_open[node_info[0]].remove((n_dist, n_ele))
        G_close[node_info[0]].append((n_dist, n_ele))
        return node_info, open_list, G_open, G_close
    else:
    # 1. Select set of paths from open_list that is not dominated
        candidate_list = [open_list[0]]
        for i in range(1, len(open_list)):
            candidate_list = prune_list(candidate_list, open_list[i])
            # If all elements in open_list are dominated, dom_node becomes the single new element
            if len(candidate_list) == 0:
                candidate_list = [open_list[i]]
        # Return the node with the smallest euclidean distance so far:
        node_info = candidate_list[0]
        for i in range(1, len(candidate_list)):
            (n, (n_dist, n_ele), (n_dist_heu, n_ele_heu)) = candidate_list[i]
            if n_dist_heu < node_info[2][1]:
                node_info = (n, (n_dist, n_ele), (n_dist_heu, n_ele_heu))
        # Remove the selected node
        open_list.remove(node_info)
        # Update: remove (n_dist, n_ele) from Gop and move to Gcl
        (n, (n_dist, n_ele), (n_dist_heu, n_ele_heu)) = node_info
        G_open[node_info[0]].remove((n_dist, n_ele))
        G_close[node_info[0]].append((n_dist, n_ele))
        return node_info, open_list, G_open, G_close

# Check if dom_node dominates any of the things in open_list, if yes, update
# Input: open_list, dom_node
# Output: the updated open_list
def prune_list(open_list, dom_node):
	# Extract info from dom_node
	(dn, (dn_dist, dn_ele), (dn_dist_heu, dn_ele_heu)) = dom_node
	# Go through each element in open_list, delete if dominated by dom_node
	to_delete = []
	for i in range(len(open_list)):
		(n, (n_dist, n_ele), (n_dist_heu, n_ele_heu)) = open_list[i]
		if check_dominance((n_dist_heu, n_ele_heu), (dn_dist_heu, dn_ele_heu)):
			to_delete.append(i)
	# Now delete the indices to be deleted
	open_list = [open_list[i] for i in range(len(open_list)) if i not in to_delete]
	return open_list

# Check if dom_heu dominates any of the things in g_list[neighbor], if yes, update
# Input: g_list (Gopen or Gclose), neighbor, dom_heu (the )
# Output: the updated open_list
def prune_list_open_close(g_list, neighbor, dom_heu):
    # Extract h1 from dom_heu
    dh1, dh2 = dom_heu
    # Go through each element in open_list, delete if dominated by dom_node
    for i in range(len(g_list[neighbor])):
        h1, h2 = g_list[neighbor][i]
        if check_dominance((h1, h2), (dh1, dh2)):
            del g_list[neighbor][i]
    # If all the things are deleted, add dom_heu to g_list[neighbor]:
    if len(g_list[neighbor]) == 0:
        g_list[neighbor].append(dom_heu)
    return g_list

# Backtracks through graph from nodes in goal_list
# Input: The recorded set of prev_nodes to the goal and the corresponding costs,
#        G_close, recorded best_costs, start node
# Output: the list of solution paths with costs in COSTS
def backtrack(GOALN, COSTS, G_close, best_costs, start):
    all_paths = []
    for i in range(len(GOALN)):
        node_cost = COSTS[i]
        prev_node = best_costs[GOALN[i]][node_cost]
        path = [prev_node, GOALN[i]]
        while prev_node != start:
            # Get the selected cost of the previous node
            temp_cost = (G_close[prev_node][0][0], G_close[prev_node][0][1]) 
            temp_node = best_costs[prev_node][temp_cost]
            path.insert(0, temp_node)
            prev_node = temp_node
        all_paths.append(path)
    return all_paths

# ------ END: Helper functions -------

# NOTES:
# Dataframe: x, y, elevation, index (aka row number) -- each row is unique
# Dictionary:
#   Input: give index
#   Output: indices of the neighbor(s)

# start: node index within the dataframe
# end: node index within the dataframe

# OUTPUT: set of vertices along the path with COSTS

# MULTIOBJECTIVE A* SEARCH
# Input: dataframe
#        start -- x,y
#        goal -- x,y
# Output: set of solution paths with costs in COSTS
def astar(dataframe, start, goal):

    # 1. Create
    # Get the coordiantes of start, goal
    start_coord = get_node_coords(nodes_df, start)
    goal_coord = get_node_coords(nodes_df, goal)
    # Initialize list of alternatives triple(n, g , F(n,g))
    open_list = [(start, \
                  (0,0), \
                  (euclidean_distance(start_coord, goal_coord), \
                   elevation(start_coord, goal_coord)))]

    # Initialize two empty sets
    GOALN = []
    COSTS = [] # potentially dictionary of nodes with costs

    # Dictionary -- mapping nodes to Gop(n) && Gcl(n)
    G_open = {start: [(0,0)]}
    G_close = {start: []}
    best_costs = {start:{(0,0):None}}

    # 2. Check Termination (skipped, at the end)
    # if OPEN is empty, then backtrack in SG from the nodes in GOALN
    #   && return the set of solution paths w/ costs in COSTS


    # 3. Path Selection
    while open_list:
        # print("G_close:", G_close)
        # Select the nondominated (n, gn, F) from open_list, breaking tie by smallest n_dist
        # Done in path_selection -- Delete (n, gn, F) from OPEN, and move gn from Gop(n) go Gcl(n)
        node_info, open_list, G_open, G_close = path_selection(open_list, G_open, G_close)
        (node, (n_dist, n_ele), (n_dist_heu, n_ele_heu)) = node_info

        # 4. Solution Recording
        # If n is goal, then
        if node == goal:
            # Include n in GOALN and gn in COSTS
            GOALN += [node]
            COSTS += [(n_dist, n_ele)]
            # Eliminate from open_list all alternatives (x, gx, Fx) 
            # such that all vectors in Fx are dominated by gn (FILTERING)
            open_list = prune_list(open_list, node_info)

        # 5. Path Expansion
        # If n is not goal, then
        else:
            # For all successors nodes m of n that do not produce cycles in SG do:
            for neighbor in return_neighbors(neighbors, node):
                # (a) Calculate the cost of the new path found to m: gm = gn + c(n,m)
                node_coord = get_node_coords(nodes_df, node)
                neighbor_coord = get_node_coords(nodes_df, neighbor)
                next_dist = n_dist + euclidean_distance(node_coord, neighbor_coord)
                next_ele = n_ele + elevation(node_coord, neighbor_coord)
                # (b) If m is a new node
                if neighbor not in G_open.keys():
                    # i. Calculate Fm = F (m, gm) filtering estimates dominated by COSTS
                    F_dist, F_ele = next_dist + euclidean_distance(neighbor_coord, goal_coord), \
                                    next_ele + elevation(neighbor_coord, goal_coord)
                    # ii. If Fm is not empty, put (m, gm, Fm) in OPEN,
                    if not any([check_dominance((F_dist, F_ele), (h1,h2)) for h1,h2 in COSTS]):
                        #  and put gm in Gop(m) labelling a pointer to n
                        open_list.append((neighbor, (next_dist, next_ele), (F_dist, F_ele)))
                        G_open[neighbor] = [(next_dist, next_ele)]
                        G_close[neighbor] = []
                        best_costs[neighbor] = {}
                        best_costs[neighbor][(next_dist, next_ele)] = node
                    # iii. Go to step 2 (skipped)
                # else (m is not a new node), in case
                else:
                    # 1. gm is in Gop(m) or gm is in Gcl(m):
                    if (next_dist, next_ele) in G_open[neighbor] or (next_dist, next_ele) in G_close[neighbor]:
                        # label with gm a pointer to n
                        best_costs[neighbor] = {}
                        best_costs[neighbor][(next_dist, next_ele)] = node
                        # Go to step 2 (skipped)
                    # TODO: 2. If gm is non-dominated by any cost vectors in Gop(m) U Gcl(m)
                    dom = True
                    for (h1,h2) in G_close[neighbor] + G_open[neighbor]:
                        # if anything is G_close[neighbor] + G_open[neighbor] dominates next_dist, next_ele,
                        # set dom = False
                        if check_dominance((next_dist, next_ele), (h1, h2)):
                            dom = False
                    # If next_dist, next_ele dominates everything, 
                    # i.e. a path to m with new cost had been found, then:
                    if dom:
                        # i. Eliminate from Gop(m) and Gcl(m) vectors dominated by gm
                        G_open = prune_list_open_close(G_open, neighbor, (next_dist, next_ele))
                        G_close = prune_list_open_close(G_close, neighbor, (next_dist, next_ele))
                        # ii. Calculate Fm = F(m, gm) filtering estimates dominated by COSTS
                        F_dist, F_ele = next_dist + euclidean_distance(neighbor_coord, goal_coord), \
                                        next_ele + elevation(neighbor_coord, goal_coord)
                        # iii. If Fm is not empty, put (m, gm, Fm) in open_list, and put gm in Gop(m) 
                        # labelling a pointer to n.
                        if not any([check_dominance((F_dist, F_ele), (h1,h2)) for h1,h2 in COSTS]):
                            #  and put gm in Gop(m) labelling a pointer to n
                            open_list.append((neighbor, (next_dist, next_ele), (F_dist, F_ele)))
                            G_open[neighbor].append((next_dist, next_ele))
                            best_costs[neighbor] = {}
                            best_costs[neighbor][(next_dist, next_ele)] = node
                        # iv. Go to step 2 (skipped)

                    # 3. Otherwise: go to step 2 (skipped)
    # return set of solutions, else return error
    if len(GOALN) > 0:
        all_paths = backtrack(GOALN, COSTS, G_close, best_costs, start)
        print("Found", len(all_paths), "paths from node", start, "to node", goal, ":")
        for i in range(len(all_paths)):
            print("Path", i, ":", all_paths[i])
        return all_paths
    else:
        print("No path found!")
        return None

# Test
astar(nodes_df, 1, 10)