import numpy as np
import time
from helper_functions import calculate_distance_matrix

def calculate_tour_cost(route, distance_matrix):
    """Calculates the total distance of a full tour."""
    if route is None or len(route) < 2:
        return float('inf')

    route_array = np.array(route)
    current_cities = route_array
    next_cities = np.roll(route_array, -1)

    # Use vectorized lookup (fast)
    distances = distance_matrix[current_cities, next_cities]
    return np.sum(distances)

def get_swap_neighbors(route):
    """Generates all neighbors reachable by a single swap operation."""
    n = len(route)
    route = list(route)  # Ensure it's mutable list
    neighbors = []

    # Generate all unique pairs (i, j) where i < j
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = route[:]  # Copy the route
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)

    return neighbors


#  Main Hill Climbing TSP Solver
def hill_climbing_tsp(cities, max_iterations=5000, start_route=None):
    """
    Implements Steepest Ascent Hill Climbing using the Swap neighborhood.

    Parameters
    ----------
    cities : np.ndarray
        City coordinates.
    max_iterations : int
        Maximum number of non-improving iterations before stopping.
    start_route : list or None
        Starting route. If None, a random route is generated.

    Returns
    -------
    (final_route, final_cost)
    """

    n = len(cities)
    distance_matrix = calculate_distance_matrix(cities)

    # Initialize with a random route if none provided
    if start_route is None:
        current_route = np.random.permutation(n).tolist()
    else:
        current_route = start_route

    current_cost = calculate_tour_cost(current_route, distance_matrix)

    non_improving_steps = 0
    total_steps = 0

    print(f"Hill Climbing: Initial Cost = {current_cost:.2f}")

    while non_improving_steps < max_iterations:
        total_steps += 1

        #Generate all neighbors
        neighbors = get_swap_neighbors(current_route)

        best_neighbor = None
        best_neighbor_cost = float('inf')

        #Find the best move
        for neighbor_route in neighbors:
            cost = calculate_tour_cost(neighbor_route, distance_matrix)

            if cost < best_neighbor_cost:
                best_neighbor_cost = cost
                best_neighbor = neighbor_route

        #Check for improvement
        if best_neighbor_cost < current_cost:
            # Move to the new best state
            current_route = best_neighbor
            current_cost = best_neighbor_cost
            non_improving_steps = 0  # Reset counter

        else:
            #Local optimum reached for the Swap neighborhood
            non_improving_steps += 1
            if non_improving_steps >= max_iterations:
                break

    print(f"Hill Climbing: Steps = {total_steps}, Final Cost = {current_cost:.2f}")
    return current_route, current_cost



def greedy_coloring(adjacency_list, num_colors):
    """
    Greedy heuristic for baseline
    :param adjacency_list:
    :param num_colors:
    :return:
    """
    n = len(adjacency_list)
    solution = np.zeros(n, dtype=int)
    for node in range(n):
        neighbor_colors = {solution[nei] for nei in adjacency_list[node]}
        for c in range(num_colors):
            if c not in neighbor_colors:
                solution[node] = c
                break
    return solution


