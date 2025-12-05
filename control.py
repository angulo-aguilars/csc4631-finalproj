import heapq
import numpy as np

def tsp_astar(distance_matrix, start=0):
    n = distance_matrix.shape[0]

    initial_visited = frozenset([start])
    start_node = (0.0, 0.0, start, initial_visited, [start])

    pq = []
    heapq.heappush(pq, start_node)

    expansions = 0

    min_g_to_state = {(start, initial_visited): 0.0}

    while pq:
        f, g, current, visited, route = heapq.heappop(pq)
        expansions += 1

        if len(visited) == n:
            total_cost = g + distance_matrix[current, start]
            final_route = route + [start]

            return final_route, total_cost

        for nxt in range(n):
            if nxt in visited:
                continue

            g2 = g + distance_matrix[current, nxt]
            visited2 = visited | {nxt}
            state2 = (nxt, visited2)

            if state2 in min_g_to_state and g2 >= min_g_to_state[state2]:
                continue

            min_g_to_state[state2] = g2

            route2 = route + [nxt]

            h2 = 0.0
            f2 = g2 + h2

            heapq.heappush(pq, (f2, g2, nxt, visited2, route2))

    return None, float("inf")