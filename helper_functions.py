"""
helper_functions.py

This file provides helper functions for both the Traveling Salesman Problem
and the Graph Coloring Problem.

Includes:
- TSP distance metrics (edge difference, Euclidean distance matrix)
- 2-opt neighborhood operator for route improvement
- GCP hamming distance metric between color assignments
- Random graph generator for GCP experiments.

"""
import numpy as np

# TSP HELPERS
def calculate_solution_distance_tsp(route1, route2):
    """
    Calculates the edge distance between two TSP tours, defined as the number of
    differing edges. This metric is used to determine solution similarity for fitness sharing
    and population diversity.

    :param route1: TSP first route
    :param route2: TSP second route
    :return: int: The count of edges that are present in one route but not the other
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
    """
    Calculates the nxn distance matrix using the Euclidean distance between all pairs of city coordinates.
    This metric is precalculated for faster fitness evaluation

    :param cities: np.ndarray of shape n x 2 containing city coordinates
    :return: np.ndarray: the symmetric distance matrix
    """

    cities_array = np.array(cities)
    diff = cities_array[:, None, :] - cities_array[None, :, :]
    distance_matrix = np.linalg.norm(diff, axis=2)

    # Ensure diagonal remains zero (distance to itself)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

def two_opt_swap(route, i, k):
    """
    Performs a 2-opt swap by reversing the segment of the route from index i to k (inclusive).

    :param route: the current route
    :param i: start index of the segment to reverse, inclusive
    :param k: end index of the segment to reverse, inclusive
    :return: np.ndarray: the new route after the 2 opt swap
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
    Computes the hamming distance between two graph-coloring assignments.

    :param individual1: First coloring assignment (list of color integers).
    :param individual2: Second coloring assignment (list of color integers).
    :return: fraction of mismatched colors.
    """
    individual1 = np.asarray(individual1)
    individual2 = np.asarray(individual2)
    return np.mean(individual1 != individual2)


def generate_random_graph(num_nodes, edge_probability=0.3):
    """
    Generates a random graph.

    :param num_nodes: Number of vertices in the graph.
    :param edge_probability: Probability of generating each possible edge.
    :return: Adjacency list representation as a dictionary.
    """
    graph = {i: [] for i in range(num_nodes)}

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_probability:
                graph[i].append(j)
                graph[j].append(i)

    return graph