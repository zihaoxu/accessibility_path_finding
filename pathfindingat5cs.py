import math, time
import numpy as np

# Unit Conversions
lat_to_m = 111 * 1000
lng_to_m = 111.3 * 1000

# Load in necessary files
nodes = pd.read_csv(mst_path + 'clean_node.csv')
with open('neighbors.pickle', 'rb') as f:
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
#        nodeCosts (Gop & Gcl from write up)
# Output: a item from open_list, updated open_list, updated nodeCosts
def path_selection(open_list, nodeCosts):
	if len(open_list) == 1:
		node_info = open_list.pop(0)
		return node_info, open_list, nodeCosts
	else:
	# 1. Select set of paths from open_list that is not dominated
		candidate_list = [open_list[0]]
		for i in range(1, len(open_list)):
			candidate_list = prune_list(candidate_list, open_list[i])
			# If all elements in open_list are dominated, dom_node becomes the single new element
			if len(open_list) == 0:
				open_list = [dom_node]
		# Return the node with the smallest euclidean distance so far:
		node_info = candidate_list[0]
		for i in range(1, len(candidate_list)):
			(n, (n_dist, n_ele), (n_dist_heu, n_ele_heu)) = candidate_list[i]
			if n_dist_heu < node_info[2][1]:
				node_info = (n, (n_dist, n_ele), (n_dist_heu, n_ele_heu))
		# Remove the selected node
		open_list.remove(node_info)
		# Update nodeCosts: remove (n_dist, n_ele) from Gop and move to Gcl
		nodeCosts[node_info[0]][0].remove((n_dist, n_ele))
		nodeCosts[node_info[0]][1].append((n_dist, n_ele))
		return node_info, open_list, nodeCosts

# Check if dom_node dominates any of the things in open_list, if yes, update
# Input: open_list, dom_node
# Output: the updated open_list
def prune_list(open_list, dom_node):
	# Extract info from dom_node
	(dn, (dn_dist, dn_ele), (dn_dist_heu, dn_ele_heu)) = dom_node
	# Go through each element in open_list, delete if dominated by dom_node
	for i in range(len(open_list)):
		(n, (n_dist, n_ele), (n_dist_heu, n_ele_heu)) = open_list[i]
		if check_dominance((n_dist_heu, n_ele_heu), (dn_dist_heu, dn_ele_heu)):
			del open_list[i]
	return open_list

# TODO: Backtracks through graph from nodes in goal_list
# Output: the set of solution paths with costs in COSTS
def backtrack(goal_list, COSTS, best_costs):
    pass

# ------ END: Helper functions -------

# NOTES:
# Dataframe: x, y, elevation, index (aka row number) -- each row is unique
# Dictionary:
#   Input: give index
#   Output: indices of the neighbor(s)

# start: x, y coordinate
# end: x, y coordinate

# OUTPUT: set of paths with COSTS and edges/nodes

# MULTIOBJECTIVE A* SEARCH
# Input: dataframe
#        start -- x,y
#        goal -- x,y
# Output: set of solution paths with costs in COSTS
def astar(dataframe, start, goal):

    # 1. Create
    # Initialize list of alternatives triple(n, g , F(n,g))
    open_list = [(start, \
    	         (0,0), \
                 (euclidean_distance(start, goal), elevation(start, goal)))]
    best_costs = {start: (0, start)}

    # Initialize two empty sets
    GOALN = []
    COSTS = [] # potentially dictionary of nodes with costs

    # Dictionary -- mapping nodes to Gop(n) && Gcl(n)
    nodeCosts = {
        start: ([(0,0)],[], None)
    }

    # 2. Check Termination (skipped, at the end)
    # if OPEN is empty, then backtrack in SG from the nodes in GOALN
    #   && return the set of solution paths w/ costs in COSTS


    # 3. Path Selection
    while open_list:
        # Select the nondominated (n, gn, F) from open_list, breaking tie by smallest n_dist
        # Done in path_selection -- Delete (n, gn, F) from OPEN, and move gn from Gop(n) go Gcl(n)
        node_info, open_list, nodeCosts = path_selection(open_list, nodeCosts)
        (node, (n_dist, n_ele), (n_dist_heu, n_ele_heu)) = node_info

        # 4. Solution Recording
        # If n is goal, then
        if n == goal:
            # Include n in GOALN and gn in COSTS
            GOALN += [n]
            COSTS += [(n_dist, n_ele)]
            # Eliminate from open_list all alternatives (x, gx, Fx) 
            # such that all vectors in Fx are dominated by gn (FILTERING)
            open_list = prune_list(open_list, node_info)

        # 5. Path Expansion
        # If n is not goal, then
        else:
            # For all successors nodes m of n that do not produce cycles in SG do:
            for neighbor in return_neighbors(node):
                # (a) Calculate the cost of the new path found to m: gm = gn + c(n,m)
                get_node_coords(node)
                get_node_coords(neighbor)
                next_dist = n_dist + euclidean_distance(node, neighbor)
                next_ele = n_ele + elevation(node, neighbor)
                # (b) If m is a new node
                if neighbor not in open_list:
                    # i. Calculate Fm = F (m, gm) filtering estimates dominated by COSTS
                    F_dist, F_ele = next_dist + euclidean_distance(neighbor, goal), next_ele + elevation(neighbor, goal)
                    # ii. If Fm is not empty, put (m, gm, Fm) in OPEN,
                    #  and put gm in Gop(m) labelling a pointer to n
                    open_list[neighbor] = (neighbor, (next_dist, next_ele), (F_dist, F_ele))
                    nodeCosts[neighbor] = ([(next_dist, next_ele)],[], node)
                    # iii. Go to step 2 (skipped)
                # else (m is not a new node), in case
                else:
                    # 1. gm is in Gop(m) or gm is in Gcl(m):
                    if (next_dist, next_ele) in nodeCosts[neighbor][0] or (next_dist, next_ele) in nodeCosts[neighbor][1]:

                        # label with gm a pointer to n
                        nodeCosts[neighbor][2] += node
                        # Go to step 2
                    # TODO: 2. If gm is non-dominated by any cost vectors in Gop(m) U Gcl(m)
                    # (a path to m with new cost had been found), then:
                        # i. Eliminate from Gop(m) and Gcl(m) vectors dominated by gm
                        # ii. Calculate Fm = F(m, gm) filtering estimates dominated by COSTS
                        # iii. If Fm is not empty, put (m, gm, Fm) in open_list, and put gm in Gop(m) labelling a pointer to n.
                        # iv. Go to step 2 (skipped)

                    # 3. Otherwise: go to step 2 (skipped)
    # TODO: Implement backtracking if goal is found -- then return set of solutions, else return error
    if len(GOALN) > 0:
        return backtrack(GOALN, COSTS, best_costs)
    else:
    	print("No path found!")
    	return None
