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
        current_cities = individual
        next_cities = np.roll(individual, -1)

        distances = self.distance_matrix[current_cities, next_cities]
        distance = np.sum(distances)

        if distance <= 1e-8:
            return 1e-10

        return 1.0 / distance

    def crossover(self, parent1, parent2):
        size = len(parent1)
        if size < 2:
            return parent1.copy()

        #Choosing segment bounds and copy segment from P1
        start, end = sorted(np.random.choice(size, 2, replace=False))

        offspring = np.full(size, -1, dtype=int)
        offspring[start:end + 1] = parent1[start:end + 1]
        copied_genes = set(parent1[start:end + 1])

        #sequence to fill the remaining gaps

        # Get cities from P2 that are NOT in the copied segment, preserving their P2 order
        fill_sequence = [
            gene for gene in parent2
            if gene not in copied_genes
        ]

        #Determine the insertion points
        #Insertion starts at (end + 1) and wraps around to 0
        insertion_points = list(range(end + 1, size)) + list(range(0, start))

        #Inserting the genes from fill_sequence into the calculated points
        for idx, gene in zip(insertion_points, fill_sequence):
            offspring[idx] = gene

        return offspring

    def mutate(self, individual):
        current_route = individual
        n = self.num_cities

        if n < 3:
            return individual

        best_route = current_route.copy()  # Start with a copy
        best_distance = self._calculate_total_distance(individual)
        improved = True

        while improved:
            improved = False
            for i in range(1, n - 1):
                for k in range(i + 1, n):
                    # Optimized edge-cost comparison

                    p_i_minus_1 = best_route[i - 1]
                    p_i = best_route[i]
                    p_k = best_route[k]
                    p_k_plus_1 = best_route[(k + 1) % n]

                    cost_removed = (self.distance_matrix[p_i_minus_1, p_i] +
                                    self.distance_matrix[p_k, p_k_plus_1])

                    cost_added = (self.distance_matrix[p_i_minus_1, p_k] +
                                  self.distance_matrix[p_i, p_k_plus_1])

                    if cost_added < cost_removed:
                        best_route = two_opt_swap(best_route, i, k)

                        best_distance = best_distance - cost_removed + cost_added
                        improved = True
                        break
                if improved:
                    break

        return best_route

    def solution_distance(self, route1, route2):
        diff_count = calculate_solution_distance_tsp(route1, route2)
        return diff_count / self.num_cities

    def _calculate_total_distance(self, individual):
        """Internal helper to calculate total distance without fitness transformation."""
        current_cities = individual
        next_cities = np.roll(individual, -1)
        distances = self.distance_matrix[current_cities, next_cities]
        return np.sum(distances)

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
                if v > u and int(individual[u]) == int(individual[v]):
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
