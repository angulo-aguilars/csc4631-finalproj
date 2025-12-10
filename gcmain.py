import numpy as np
import matplotlib.pyplot as plt
from problem import GCPProblem
from agent import Agent
from helper_functions import generate_random_graph

# ----------------------------
# Experiment Parameters
# ----------------------------
num_instances = 5
num_trials = 10
num_nodes = 20
num_colors = 4
population_size = 20
max_generations = 50


# ----------------------------
# Baseline: Greedy Coloring
# ----------------------------
def greedy_coloring(graph, num_colors):
    solution = np.full(len(graph), -1, dtype=int)
    for v in range(len(graph)):
        forbidden = {solution[u] for u in graph[v] if solution[u] != -1}
        for color in range(num_colors):
            if color not in forbidden:
                solution[v] = color
                break
    # Fitness = negative number of conflicts
    conflicts = 0
    for u in range(len(graph)):
        for v in graph[u]:
            if v > u and solution[u] == solution[v]:
                conflicts += 1
    return solution, -conflicts


# ----------------------------
# Generate Random Instances
# ----------------------------
instances = [generate_random_graph(num_nodes, edge_probability=0.2) for _ in range(num_instances)]

# ----------------------------
# Run Experiments
# ----------------------------
results = {
    'advanced_ga': [],
    'simple_ga': [],
    'greedy': []
}

for idx, graph in enumerate(instances):
    print(f"\n--- Instance {idx} ---")
    problem = GCPProblem(graph, num_colors)

    adv_trial_best = []
    sim_trial_best = []
    greedy_best, _ = greedy_coloring(graph, num_colors)

    for trial in range(num_trials):
        # --- Advanced GA ---
        adv_agent = Agent(problem, population_size=population_size, max_generations=max_generations)
        adv_agent.run_evolution()
        _, adv_fitness = adv_agent.get_best_solution()
        adv_trial_best.append(adv_fitness)

        # --- Simple GA ---
        sim_agent = Agent(problem, population_size=population_size, max_generations=max_generations)
        # Disable adaptive mutation / sharing / steady-state for simple GA if needed
        # For simplicity, we just use same agent with smaller population
        sim_agent.run_evolution()
        _, sim_fitness = sim_agent.get_best_solution()
        sim_trial_best.append(sim_fitness)

        print(f"Trial {trial} | Adv GA: {adv_fitness:.1f}, Simple GA: {sim_fitness:.1f}")

    # Store results
    results['advanced_ga'].append(adv_trial_best)
    results['simple_ga'].append(sim_trial_best)
    # Greedy is deterministic
    results['greedy'].append([_ for _ in range(num_trials)])


# ----------------------------
# Summarize Results
# ----------------------------
def summarize(trial_data):
    trial_data = np.array(trial_data)
    avg = np.mean(trial_data)
    std = np.std(trial_data)
    return avg, std


print("\n--- Summary of Best Fitness per Instance ---")
for idx in range(num_instances):
    adv_avg, adv_std = summarize(results['advanced_ga'][idx])
    sim_avg, sim_std = summarize(results['simple_ga'][idx])
    greedy_avg, greedy_std = summarize(results['greedy'][idx])
    print(f"Instance {idx} | Advanced GA: {adv_avg:.2f} ± {adv_std:.2f}, "
          f"Simple GA: {sim_avg:.2f} ± {sim_std:.2f}, "
          f"Greedy: {greedy_avg:.2f}")

# ----------------------------
# Plot Convergence for first instance
# ----------------------------
plt.figure(figsize=(10, 5))
adv_agent = Agent(GCPProblem(instances[0], num_colors), population_size=population_size,
                  max_generations=max_generations)
adv_agent.run_evolution()
plt.plot(adv_agent.best_fitness_history, label="Advanced GA")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Convergence Example (Instance 0)")
plt.legend()
plt.show()
