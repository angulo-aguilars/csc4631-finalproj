import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from agent import Agent
from problem import TSPProblem
from control import tsp_astar

POP_SIZE = 100
MAX_GEN = 500
NUM_CITIES_TSP = 20
SHARING_RADIUS = 0.3
RESULTS_DIR = "tsp_results"



def run_tsp_comparison():
    """
    runs the TSP GA vs A* Control
    """

    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # Generate random TSP instance
    cities = np.random.rand(NUM_CITIES_TSP, 2) * 100
    tsp_problem = TSPProblem(cities)

    # run ga
    print(f"\nRunning GA on {NUM_CITIES_TSP} cities...")

    start_time_ga = time.time()
    ga_agent = Agent(
        tsp_problem,
        population_size=POP_SIZE,
        max_generations=MAX_GEN,
        sharing_radius=SHARING_RADIUS,
        tournament_size=5
    )
    ga_agent.run_evolution()
    ga_time = time.time() - start_time_ga

    best_route_ga, best_fitness_ga = ga_agent.get_best_solution()
    best_cost_ga = 1.0 / best_fitness_ga

    # run a* search
    print(f"\nRunning A* Search on {NUM_CITIES_TSP} cities...")

    start_time_astar = time.time()
    astar_route, astar_cost = tsp_astar(
        tsp_problem.distance_matrix,
    )
    astar_time = time.time() - start_time_astar

    # print results
    print("\nResults: TSP")
    print(f"A* Cost : {astar_cost:.2f} | Time: {astar_time:.2f}s")
    print(f"GA Cost: {best_cost_ga:.2f} | Time: {ga_time:.2f}s")

    if 0 < astar_cost < float('inf'):
        error = ((best_cost_ga - astar_cost) / astar_cost) * 100
        print(f"GA Optimization Error: {error:.2f}% relative to optimal")
        optimal_fitness = 1.0 / astar_cost
    else:
        print("A* could not complete — no optimal reference available.")
        optimal_fitness = 0

   #plot for fitness eval
    plt.figure(figsize=(12, 5))
    plt.plot(ga_agent.best_fitness_history, label="GA Best Fitness (1/Cost)")

    if optimal_fitness > 0:
        plt.axhline(
            y=optimal_fitness,
            color="r",
            linestyle="--",
            label=f"A* Optimal Fitness ({optimal_fitness:.4f})"
        )

    plt.title("TSP: Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (1/Distance)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "tsp_fitness_over_time.png"))
    plt.show()
    plt.close()

    #plot for final cost compatison
    plt.figure(figsize=(8, 6))
    costs = [best_cost_ga]
    labels = ["Advanced GA"]

    if astar_cost < float("inf"):
        costs.append(astar_cost)
        labels.append("A* (Optimal)")

    plt.bar(labels, costs, color=["darkgreen", "red"][:len(labels)])
    plt.title(f"TSP Final Cost Comparison (N={NUM_CITIES_TSP})")
    plt.ylabel("Total Distance")
    plt.savefig(os.path.join(RESULTS_DIR, "tsp_cost_comparison.png"))
    plt.show()
    plt.close()

if __name__ == "__main__":
    run_tsp_comparison()

