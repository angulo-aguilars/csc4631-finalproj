import numpy as np
from problem import GCPProblem
from agent import Agent
from helper_functions import generate_random_graph

# -------------------------
# 1. Generate random graph
# -------------------------
num_nodes = 10
num_colors = 4

graph = generate_random_graph(num_nodes, edge_probability=0.3)

# -------------------------
# 2. Create GCP problem
# -------------------------
problem = GCPProblem(graph, num_colors)

# -------------------------
# 3. Create the Genetic Agent
# -------------------------
agent = Agent(
    problem=problem,
    population_size=50,
    max_generations=200
)

# -------------------------
# 4. Run Evolution
# -------------------------
final_population = agent.run_evolution()

# -------------------------
# 5. Get best solution
# -------------------------
best_solution, best_fitness = agent.get_best_solution()

print("Best Coloring:", best_solution)
print("Conflicts:", -best_fitness)
