"""
gcmain.py

This file runs controlled experiments on the Graph Coloring Problem (GCP).
It compares the performance of an advanced evolutionary Agent
(Genetic Algorithm) against a deterministic greedy-coloring baseline algorithm.

The purpose of this file:
- Generate random graphs of varying size and edge density.
- Evaluate the Agent’s evolutionary search performance across many trials.
- Collect fitness histories and population diversity metrics.
- Compute baseline greedy solutions for comparison.
• Produce summary visualizations: mean ± std fitness, convergence plots,
  and diversity evolution plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from problem import GCPProblem
from agent import Agent
from helper_functions import generate_random_graph
from control import greedy_coloring


def run_graph_coloring_experiments(num_vertices_list=[100],
                                   edge_densities=[0.1,0.3,0.5],
                                   trials=1,
                                   num_colors=None,
                                   agent_params=None):
    """
    Run all graph-coloring experiments using both the
    evolutionary Agent and a greedy heuristic baseline.

    :param num_vertices_list: List of graph sizes to evaluate.
    :param edge_densities: List of edge probabilities for random graphs.
    :param trials: Number of random graph trials per setting.
    :param num_colors: Optional fixed number of colors for all runs.
    :param agent_params: Dictionary of GA hyperparameters.
    :return: Dictionary of experiment results including fitness and histories.
    """
    results = {}

    if agent_params is None:
        agent_params = {
            'population_size': 50,
            'max_generations': 1000,
            'initial_mutation_rate': 0.1,
            'tournament_size': 5,
            'sharing_radius': 0.3,
            'replacement_count': 2,
            'maximize': True
        }

    for n in num_vertices_list:
        for p in edge_densities:
            key = f"n={n}_p={p}"
            results[key] = {'agent': [], 'greedy': [],
                            'fitness_histories': [], 'diversity_histories': []}
            print(f"Running experiments for {key}...")

            for _ in range(trials):
                adjacency_list = generate_random_graph(n, p)

                k = num_colors if num_colors else max(1, int(n*p*1.5))

                # Initialize problem
                problem = GCPProblem(adjacency_list, k)

                # Greedy baseline
                greedy_sol = greedy_coloring(adjacency_list, k)
                results[key]['greedy'].append(problem.evaluate_fitness(greedy_sol))

                # Advanced GA using Agent
                agent = Agent(problem, **agent_params)
                agent.run_evolution()
                best_sol, best_fit = agent.get_best_solution()

                results[key]['agent'].append(best_fit)
                results[key]['fitness_histories'].append(agent.best_fitness_history)
                results[key]['diversity_histories'].append(agent.diversity_history)

    return results


def plot_results(results):
    """
    Plots bar charts, fitness convergence curves, and diversity evolution
    for the graph coloring experiment results.

    :param results: Dictionary returned from run_graph_coloring_experiments().
    :return: None
    """
    for key, val in results.items():
        # Bar plot for mean ± std
        plt.figure(figsize=(8,5))
        labels = ['Agent','Greedy']
        means = [np.mean(val['agent']), np.mean(val['greedy'])]
        stds = [np.std(val['agent']), np.std(val['greedy'])]

        plt.bar(labels, means, yerr=stds, capsize=5)
        plt.ylabel('Fitness (higher is better, i.e., fewer conflicts)')
        plt.title(f'Graph Coloring Results for {key}')
        plt.show()

        # Plot fitness over generations (average)
        if val['fitness_histories']:
            fitness_array = np.array(val['fitness_histories'])
            avg_fitness = np.mean(fitness_array, axis=0)
            plt.figure(figsize=(8,5))
            plt.plot(avg_fitness, label='Average Agent Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title(f'Fitness Convergence for {key}')
            plt.legend()
            plt.show()

        # Plot diversity over generations (average)
        if val['diversity_histories']:
            diversity_array = np.array(val['diversity_histories'])
            avg_diversity = np.mean(diversity_array, axis=0)
            plt.figure(figsize=(8,5))
            plt.plot(avg_diversity, label='Average Population Diversity')
            plt.xlabel('Generation')
            plt.ylabel('Diversity')
            plt.title(f'Population Diversity Over Generations for {key}')
            plt.legend()
            plt.show()


if __name__ == "__main__":
    """
    Runs the full graph-coloring experiment with default parameters
    and generate all associated plots.

    :return: None
    """
    results = run_graph_coloring_experiments()
    plot_results(results)
