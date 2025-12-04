import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from agent import Agent
from problem import TSPProblem
from control import astar_search_tsp

POP_SIZE = 50
MAX_GEN = 200
NUM_CITIES_TSP = 7
SHARING_RADIUS = NUM_CITIES_TSP * 0.4
RESULTS_DIR = "tsp_results"
A_STAR_TIMEOUT = 120


def run_tsp_comparison():
    """
    Runs the TSP GA vs A* Control.

    This function demonstrates the trade-off between the GA's quick, near-optimal
    solution and the A* algorithm's slow, guaranteed optimal solution.
    """

    Path(RESULTS_DIR).mkdir(exist_ok=True)
    print(f"Results will be saved to the '{RESULTS_DIR}' directory.")

    print("--- Running TSP Comparison (Advanced GA vs. A* Optimal Control) ---")

    # Generate random city coordinates
    cities = np.random.rand(NUM_CITIES_TSP, 2) * 100
    tsp_problem = TSPProblem(cities)

    # 1. Advanced GA Run
    start_time_ga = time.time()
    ga_agent = Agent(tsp_problem, population_size=POP_SIZE, max_generations=MAX_GEN,
                       sharing_radius=SHARING_RADIUS)
    ga_agent.run_evolution()
    ga_time = time.time() - start_time_ga
    best_route_ga, best_fitness_ga = ga_agent.get_best_solution()
    best_cost_ga = 1.0 / best_fitness_ga

    # 2. A* Optimal Control Run
    print(f"Running A* Search on {NUM_CITIES_TSP} cities...")
    start_time_astar = time.time()
    # A* should now complete well within the 120s limit
    astar_route, astar_cost = astar_search_tsp(cities, max_time_seconds=A_STAR_TIMEOUT)
    astar_time = time.time() - start_time_astar

    print("\n--- RESULTS: TSP ---")
    print(f"A* (Optimal) Cost: {astar_cost:.2f} | Time: {astar_time:.2f}s")
    print(f"Advanced GA Cost: {best_cost_ga:.2f} | Time: {ga_time:.2f}s")

    if astar_cost < float('inf') and astar_cost > 0:
        print(f"GA Optimization Error: {((best_cost_ga - astar_cost) / astar_cost) * 100:.2f}% (relative to optimal)")
        optimal_fitness = 1.0 / astar_cost
    else:
        print("A* could not verify optimal solution within the time limit. (Cost comparison may be inaccurate)")
        optimal_fitness = 0.0

    # Plot 1: Fitness Over Evolutionary Time (Search Efficiency)
    plt.figure(figsize=(12, 5))
    plt.plot(ga_agent.best_fitness_history, label='Advanced GA Best Fitness (1/Cost)')
    if optimal_fitness > 0:
        plt.axhline(y=optimal_fitness, color='r', linestyle='--', label=f'A* Optimal Fitness ({optimal_fitness:.4f})')
    plt.title('TSP: Fitness over Evolutionary Time (Advanced GA vs. A*)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (1/Distance)')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(RESULTS_DIR, 'tsp_fitness_over_time.png'))
    plt.show()
    plt.close()

    # Plot 2: Optimization/Cost Comparison
    plt.figure(figsize=(8, 6))
    costs = [best_cost_ga]
    labels = ['Advanced GA (Heuristic)']
    if astar_cost < float('inf'):
        costs.append(astar_cost)
        labels.append('A* (Optimal)')

    plt.bar(labels, costs, color=['darkgreen', 'red'][:len(labels)])
    plt.title(f'TSP Final Route Cost Comparison (N={NUM_CITIES_TSP})')
    plt.ylabel('Total Distance Cost')
    plt.savefig(os.path.join(RESULTS_DIR, 'tsp_cost_comparison.png'))
    plt.show()
    plt.close()


if __name__ == "__main__":
    run_tsp_comparison()