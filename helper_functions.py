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

def two_opt_swap(route, i, k):
    """
    Performs a 2-opt swap by reversing the segment of the route from index i to k (inclusive).
    """
    segment_start = route[:i]

    # Segment from i to k (inclusive), reversed
    # Slicing route[i:k + 1] gets the segment, [::-1] reverses it.
    segment_reversed = route[i:k + 1][::-1]

    # Segment after k (from k+1 to end)
    segment_end = route[k + 1:]

    new_route = np.concatenate((segment_start, segment_reversed, segment_end))
    return new_route


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
    Returns an adjacency LIST (dict)
    """
    graph = {i: [] for i in range(num_nodes)}

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_probability:
                graph[i].append(j)
                graph[j].append(i)

    return graph