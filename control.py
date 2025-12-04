import heapq
import time
from helper_functions import calculate_distance_matrix, calculate_mst_cost


def astar_search_tsp(cities, max_time_seconds=120):
    """
    Implements the A* search algorithm for the TSP, optimized for memory
    by only storing the minimum state and including a robust timeout.

    State representation: (current_city, frozenset(unvisited_cities))
    """
    start_time = time.time()
    num_cities = len(cities)
    dist_matrix = calculate_distance_matrix(cities)

    # State Key: (current_city, unvisited_set)
    # Priority queue item: (f_score, g_score, current_city, unvisited_set)

    start_city = 0
    # The set of cities to visit (excluding the start city 0)
    unvisited_initial = frozenset(range(1, num_cities))

    # Heuristic: MST cost of remaining nodes (all cities plus the return trip)
    h_score_start = calculate_mst_cost(list(unvisited_initial) + [start_city], dist_matrix)
    f_score = h_score_start

    open_set = [(f_score, 0, start_city, unvisited_initial)]

    best_path_cost = float('inf')

    # g_scores tracks the minimum cost found to reach a state
    g_scores = {(start_city, unvisited_initial): 0}
    expansions = 0

    while open_set:
        expansions += 1

        if expansions % 500 == 0:
            if time.time() - start_time > max_time_seconds:
                print(f"\n[A* Search Timeout] Exceeded {max_time_seconds}s limit after {expansions} expansions.")
                return None, best_path_cost # Return best cost found so far

        f, g, current, unvisited_set = heapq.heappop(open_set)

        # Skip state if a cheaper path to this state has already been found and processed
        if g > g_scores.get((current, unvisited_set), float('inf')):
             continue

        if not unvisited_set:
            # All cities visited. Calculate final cost (current -> start_city 0)
            final_cost = g + dist_matrix[current, start_city]
            if final_cost < best_path_cost:
                best_path_cost = final_cost
            continue

        for next_city in unvisited_set:
            new_g = g + dist_matrix[current, next_city]
            new_unvisited = unvisited_set - {next_city}

            #Check against the best complete path found so far (Alpha-Pruning effect)
            if new_g >= best_path_cost:
                continue

            # Heuristic calculation: MST cost of remaining nodes + start city
            remaining_nodes = list(new_unvisited) + [next_city, start_city]
            h_score = calculate_mst_cost(remaining_nodes, dist_matrix)

            new_f = new_g + h_score
            new_state = (next_city, new_unvisited)

            # Check if this new path is better than the currently known shortest path to new_state
            if new_g < g_scores.get(new_state, float('inf')):

                # Update g_scores and push to open_set
                g_scores[new_state] = new_g
                heapq.heappush(open_set, (new_f, new_g, next_city, new_unvisited))

    print(f"\n[A* Search Complete] Total expansions: {expansions}")
    return None, best_path_cost