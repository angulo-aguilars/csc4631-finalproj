import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def calculate_distance_matrix(cities):
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i,j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
    return distance_matrix

def calculate_mst_cost(nodes_subset, distance_matrix):
    ## this method will be used to calculate the cost of the min spanning tree
    ## and used as an admissible heuristic for remaining path cost in tsp with A*
    return

def calculate_solution_distance_tsp(individual1, individual2):
    num_cities = len(individual1)
    diff_count = 0
    edges1 = set()
    for i in range(num_cities):
        city_a = individual1[i]
        city_b = individual1[(i + 1) % num_cities]
        edges1.add(tuple(sorted((city_a, city_b))))

    for i in range(num_cities):
        city_a = individual2[i]
        city_b = individual2[(i + 1) % num_cities]
        if tuple(sorted((city_a, city_b))) not in edges1:
            diff_count+=1
    return diff_count

def calculate_solution_distance_gcp():
    return

def generate_random_graph(num_nodes, edge_probability=0.3):
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_probability:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    return adj_matrix