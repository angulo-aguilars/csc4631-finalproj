import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from agent import Agent
from problem import TSPProblem
from control import hill_climbing_tsp

# Define a set of experiments to run
EXPERIMENTS = [
    {
        "name": "Experiment 1",
        "num_cities": 10,
        "pop_size": 50,
        "max_gen": 75,
        "sharing_radius": 0.3
    },
    {
        "name": "Experiment 2",
        "num_cities": 20,
        "pop_size": 100,
        "max_gen": 200,
        "sharing_radius": 0.3
    },
    {
        "name": "Experiment 3",
        "num_cities": 40,
        "pop_size": 250,
        "max_gen": 500,
        "sharing_radius": 0.3
    },
]

#Global Settings
RESULTS_BASE_DIR = "tsp_results"
TOURNAMENT_SIZE = 5


def run_single_experiment(experiment_config):
    """
    Runs the GA vs Hill Climbing comparison for a single configuration.
    Saves results into a subdirectory named after the experiment.
    """
    name = experiment_config["name"]
    NUM_CITIES_TSP = experiment_config["num_cities"]
    POP_SIZE = experiment_config["pop_size"]
    MAX_GEN = experiment_config["max_gen"]
    SHARING_RADIUS = experiment_config["sharing_radius"]

    # Create a specific directory for this experiment's results
    RESULTS_DIR = Path(RESULTS_BASE_DIR) / name
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Running Experiment: {name} (Cities: {NUM_CITIES_TSP}, Pop: {POP_SIZE}, Gen: {MAX_GEN}) ---")

    # Generate random TSP instance (The cities must be the same for both algorithms)
    cities = np.random.rand(NUM_CITIES_TSP, 2) * 100
    tsp_problem = TSPProblem(cities)

    #Run ga
    print(f"\nRunning GA...")

    start_time_ga = time.time()
    ga_agent = Agent(
        tsp_problem,
        population_size=POP_SIZE,
        max_generations=MAX_GEN,
        sharing_radius=SHARING_RADIUS,
        tournament_size=TOURNAMENT_SIZE
    )
    ga_agent.run_evolution()
    ga_time = time.time() - start_time_ga

    best_route_ga, best_fitness_ga = ga_agent.get_best_solution()
    best_cost_ga = 1.0 / best_fitness_ga if best_fitness_ga > 0 else float('inf')

    # run hc control
    print(f"\nRunning Hill Climbing...")

    start_time_hc = time.time()
    # Using MAX_GEN as max_iterations for Hill Climbing for a rough time comparison
    hc_route, hc_cost = hill_climbing_tsp(
        cities=cities,
        max_iterations=MAX_GEN
    )
    hc_time = time.time() - start_time_hc

    # Print and Calculate Results
    print("\n--- Summary ---")
    print(f"HC Cost : {hc_cost:.2f} | Time: {hc_time:.2f}s")
    print(f"GA Cost: {best_cost_ga:.2f} | Time: {ga_time:.2f}s")

    optimal_fitness = 0
    if 0 < hc_cost < float('inf'):
        error = ((best_cost_ga - hc_cost) / hc_cost) * 100
        print(f"GA Optimization Error: {error:.2f}% relative to HC control")
        optimal_fitness = 1.0 / hc_cost
    else:
        print("HC could not find a valid solution.")

    # Plot Fitness Evaluation
    plot_fitness_evaluation(ga_agent.best_fitness_history, optimal_fitness, NUM_CITIES_TSP, RESULTS_DIR)

    #Plot Final Cost Comparison
    plot_cost_comparison(best_cost_ga, hc_cost, NUM_CITIES_TSP, RESULTS_DIR)

    # Plot the Best Route Found by GA
    plot_best_route(cities, best_route_ga, "GA Best Route", "ga_best_route.png", RESULTS_DIR)

    #  Plot the Best Route Found by HC
    plot_best_route(cities, hc_route, "HC Best Route", "hc_best_route.png", RESULTS_DIR)


def plot_fitness_evaluation(ga_fitness_history, hc_fitness, num_cities, results_dir):
    """Plots the GA's best fitness over generations against the HC fitness."""
    plt.figure(figsize=(12, 5))
    plt.plot(ga_fitness_history, label="GA Best Fitness (1/Cost)")

    if hc_fitness > 0:
        plt.axhline(
            y=hc_fitness,
            color="r",
            linestyle="--",
            label=f"HC Fitness ({hc_fitness:.4f})"
        )

    plt.title(f"TSP Fitness Over Generations (N={num_cities})")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (1/Distance)")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "tsp_fitness_over_time.png")
    plt.close()


def plot_cost_comparison(ga_cost, hc_cost, num_cities, results_dir):
    """Plots the final total distance costs side-by-side."""
    plt.figure(figsize=(8, 6))
    costs = [ga_cost]
    labels = ["Advanced GA"]

    if hc_cost < float("inf"):
        costs.append(hc_cost)
        labels.append("Hill Climbing")

    plt.bar(labels, costs, color=["darkgreen", "red"][:len(labels)])
    plt.title(f"TSP Final Cost Comparison (N={num_cities})")
    plt.ylabel("Total Distance")
    plt.savefig(results_dir / "tsp_cost_comparison.png")
    plt.close()


def plot_best_route(cities, route, title, filename, results_dir):
    """Plots the best route found by an algorithm."""
    plt.figure(figsize=(8, 8))

    # Ensure the route is closed for plotting (first city is repeated at the end)
    route_closed = np.append(route, route[0])

    # Get city coordinates in the correct order
    route_coords = cities[route_closed]

    # Plot the path
    plt.plot(route_coords[:, 0], route_coords[:, 1], 'bo-', linewidth=2, markersize=8)

    # Plot the individual cities
    plt.plot(cities[:, 0], cities[:, 1], 'ro', markersize=5)

    # Label the starting city
    plt.plot(cities[route[0], 0], cities[route[0], 1], 'go', markersize=10, label='Start City')

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(results_dir / filename)
    plt.close()

def run_all_tsp_experiments():
    """Runs all defined experiments sequentially."""
    Path(RESULTS_BASE_DIR).mkdir(exist_ok=True)

    for config in EXPERIMENTS:
        run_single_experiment(config)


if __name__ == "__main__":
    run_all_tsp_experiments()