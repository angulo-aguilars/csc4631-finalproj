import numpy as np

# TSP HELPERS
def calculate_solution_distance_tsp(route1, route2):
    """
    Edge-based distance between two TSP tours.
    Returns number of differing edges.
    """
    n = len(route1)

    # Build edge set for route1
    edges_route1 = {
        tuple(sorted((route1[i], route1[(i + 1) % n])))
        for i in range(n)
    }

    diff_count = 0
    for i in range(n):
        edge = tuple(sorted((route2[i], route2[(i + 1) % n])))
        if edge not in edges_route1:
            diff_count += 1

    return diff_count

def calculate_distance_matrix(cities):
    cities_array = np.array(cities)
    diff = cities_array[:, None, :] - cities_array[None, :, :]
    distance_matrix = np.linalg.norm(diff, axis=2)

    # Ensure diagonal remains zero (distance to itself)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


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