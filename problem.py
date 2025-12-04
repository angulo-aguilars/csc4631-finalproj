from abc import ABC, abstractmethod
from helper_functions import *


class Problem(ABC):
    @abstractmethod
    def generate_random_solution(self):
        pass

    @abstractmethod
    def evaluate_fitness(self, individual):
        pass

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

    @abstractmethod
    def mutate(self, individual):
        pass

    # must match Agent usage: solution_distance(...)
    @abstractmethod
    def solution_distance(self, individual1, individual2):
        pass

    # must accept the current population and return a scalar diversity measure
    def calculate_diversity(self, population):
        distances = []
        n = len(population)
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(self.solution_distance(population[i], population[j]))
        return float(np.mean(distances)) if distances else 0.0


class TSPProblem(Problem):
    def __init__(self, cities):
        """
        cities: array-like of shape (num_cities, 2) or similar coordinates
        """
        self.cities = np.asarray(cities)
        self.num_cities = len(self.cities)
        self.distance_matrix = calculate_distance_matrix(self.cities)

    def generate_random_solution(self):
        return np.random.permutation(self.num_cities)

    def evaluate_fitness(self, individual):
        # sum tour length (wrap-around)
        distance = 0.0
        for i in range(self.num_cities):
            initial_city = individual[i]
            last_city = individual[(i + 1) % self.num_cities]
            distance += self.distance_matrix[initial_city][last_city]
        return 1.0 / distance

    def crossover(self, parent1, parent2):
        """
        PMX-like crossover. We choose start,end inclusive and fill remaining genes via mapping.
        parent1/parent2 are permutations (numpy arrays)
        """
        size = len(parent1)
        offspring = np.full(size, -1, dtype=int)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        offspring[start:end] = parent1[start:end]
        mapping = {parent2[i]: parent1[i] for i in range(start, end + 1)}
        for i in range(size):
            if offspring[i] == -1:
                gene = parent2[i]
                while gene in mapping:
                    gene = mapping[gene]
                offspring[i] = gene
        return offspring

    def mutate(self, individual):
        mutated_individual = np.copy(individual)
        idx1, idx2 = np.random.choice(self.num_cities, 2, replace=False)
        mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]
        return mutated_individual

    def solution_distance(self, individual1, individual2):
        return calculate_solution_distance_tsp(individual1, individual2)


class GCPProblem(Problem):
    def __init__(self, graph, num_colors):
        """
        graph: adjacency list or dict mapping vertex -> iterable of neighbors
        num_colors: integer
        """
        # Normalize graph to adjacency list dict of lists
        if isinstance(graph, dict):
            self.graph = {int(k): list(v) for k, v in graph.items()}
        else:
            # If graph is provided as adjacency matrix or list-of-lists, try to normalize
            self.graph = {i: list(neigh) for i, neigh in enumerate(graph)}
        self.num_vertices = len(self.graph)
        self.num_colors = int(num_colors)

    def generate_random_solution(self):
        return np.random.randint(0, self.num_colors, self.num_vertices)

    def evaluate_fitness(self, individual):
        # conflicts: count each conflicting edge once
        conflicts = 0
        for u in range(self.num_vertices):
            for v in self.graph[u]:
                if int(individual[u]) == int(individual[v]):
                    conflicts += 1
        # If adjacency lists are symmetric, each conflict counted twice; it's OK as relative measure.
        # We return negative conflicts so higher fitness == better (Agent uses argmax).
        return -float(conflicts)

    def crossover(self, parent1, parent2):
        n = self.num_vertices
        if n < 2:
            return parent1.copy()
        point = np.random.randint(1, n)  # split point in [1, n-1]
        child = np.concatenate([parent1[:point].copy(), parent2[point:].copy()])
        return child

    def mutate(self, individual):
        mutated = np.copy(individual)
        v = np.random.randint(self.num_vertices)
        current = int(mutated[v])
        choices = [c for c in range(self.num_colors) if c != current]
        if choices:
            mutated[v] = np.random.choice(choices)
        return mutated

    def solution_distance(self, ind1, ind2):
        # Hamming distance fraction in [0,1]
        arr1 = np.asarray(ind1)
        arr2 = np.asarray(ind2)
        return float(np.mean(arr1 != arr2))
