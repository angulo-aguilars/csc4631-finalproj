Evaluating Advanced Genetic Algorithms 
CSC 4631 - Final Project
Mahima Chokshi & Sandra Angulo-Aguilar

Project Overview
This project implements an Advanced Genetic algorithm agent that is capable of solving two classing NP-hard optimization problems:
The Traveling Salesperson Problem and Graph Coloring Problem. The agent is designed with four advanced mechanisms, including fitness 
sharing for diversity preservation, adaptive mutation to balance exploitation/exploration, and tournament selection with steady-state 
replacement. Performance is evaluated by comparing the GA agent's solutions against established baseline heuristics: Steepest-Ascent
Hill Climbing for TSP and Greedy Coloring algorithm for GCP.

Project Structure & File Descriptions
- agent.py: Implements the core genetic algorithm agent, managing the evolutionary loop with the four implemented advanced mechanisms
- problem.py: Defines the abstract Problem class and concrete subclasses, TSPProblem and GCPProblem, to interface optimization problems with the GA agent.
- helper_functions.py: Contains utility functions for problem specific calculations, such as TSP distance matrices, edge based distance, the 2 opt swap, and graph generation.
- control.py: implements the deterministic baseline algorithms: Steepest-Ascent Hill Climbing for TSP and the Greedy Coloring heuristic for GCP comparison
- tspmain.py: Executes a set of comparative experiments between the GA agent and Hill Climbing for the Traveling Salesperson Problem
- gcmain.py: Executes a set of comparative experiments between the GA agent and the Greedy Heuristic for the Graph Coloring Problem.
- tsp_results/: Output folder containing the plots for the Traveling Salesperson Problem experiments
- gcp_results/: Output folder containing the plots for the Graph Coloring Problem experiments.

  Instructions for Compiling and Running the Code
  Prerequisites:
  - python interpreter
  - required libraries: NumPy, Matplotlib
  - Standard Library Modules: time, os, pathlib, abc

  Running Experiments:
  1. Run Traveling Salesperson Problem Experiments:
     python tspmain.py
     This script runs the three defined TSP experiments, comparing the advanced GA against Steepest-Ascent Hill Climbing
  3. Run Graph Coloring Problem Experiments:
     python gcmain.py
     This script runs the GCP experiments across varying graph sizes and edge densities, comparing the advanced GA against the Greedy Coloring baseline.

Results and Output Interpretation 
tsp_results/
-   Content: fitness convergence plots, cost comparison bar charts, best route plots
-   Interpretation: Shows the GA's convergence to a high quality solution relative to the local opimum found by Hill Climbing and visualizes the final paths.

gcp_results/
-  Content: mean fitness bar plotos, fitness convergence plots, and diversity evolution for each configuration
-  Interpretation: Compares the mean final fitness of the agent vs. Greedy baseline, and illustrates how the agent's population evolves in terms of fitness improvement and solution diversity.
