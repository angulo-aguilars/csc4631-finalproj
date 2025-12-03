import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

# TSP HELPERS

def calculate_distance_matrix(cities):
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(
                    np.array(cities[i]) - np.array(cities[j])
                )
    return distance_matrix


def calculate_mst_cost(nodes_subset, distance_matrix):
    """
    Computes the cost of the Minimum Spanning Tree over the
    unvisited nodes for use as an admissible A* heuristic.

    nodes_subset: list of remaining city indices
    distance_matrix: full TSP distance matrix
    """
    if len(nodes_subset) <= 1:
        return 0.0

    # Build sub-matrix
    sub_matrix = distance_matrix[np.ix_(nodes_subset, nodes_subset)]

    mst = minimum_spanning_tree(sub_matrix)
    return mst.sum()


def calculate_solution_distance_tsp(individual1, individual2):
    """
    Edge-based distance between two TSP tours.
    Returns number of differing edges.
    """
    num_cities = len(individual1)

    edges1 = set()
    for i in range(num_cities):
        a = int(individual1[i])
        b = int(individual1[(i + 1) % num_cities])
        edges1.add(tuple(sorted((a, b))))

    diff_count = 0
    for i in range(num_cities):
        a = int(individual2[i])
        b = int(individual2[(i + 1) % num_cities])
        if tuple(sorted((a, b))) not in edges1:
            diff_count += 1

    # normalized distance in [0,1]
    return diff_count / num_cities


# GRAPH COLORING HELPERS

def calculate_solution_distance_gcp(individual1, individual2):
    """
    Hamming distance between two colorings.
    """
    individual1 = np.asarray(individual1)
    individual2 = np.asarray(individual2)
    return np.mean(individual1 != individual2)


def generate_random_graph(num_nodes, edge_probability=0.3):
    """
    Returns an adjacency LIST (dict), not a matrix,
    so it matches your GCPProblem class.
    """
    graph = {i: [] for i in range(num_nodes)}

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_probability:
                graph[i].append(j)
                graph[j].append(i)

    return graph