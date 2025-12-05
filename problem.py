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
        # Ensure it's a valid permutation (no duplicates/missing cities)
        if len(set(individual)) != self.num_cities:
            # Assign a heavy penalty (very low fitness value)
            return 1.0 / 10000000.0

        # sum tour length (wrap-around)
        distance = 0.0
        for i in range(self.num_cities):
            initial_city = individual[i]
            last_city = individual[(i + 1) % self.num_cities]
            distance += self.distance_matrix[initial_city][last_city]

        # Prevent division by zero/near-zero
        if distance < 1e-8:
            return 1.0 / 10000000.0

        return 1.0 / distance

    def crossover(self, parent1, parent2):
        size = len(parent1)
        if size < 2:
            return parent1.copy()

        # Convert parents to lists of standard Python ints
        p1_list = [int(g) for g in parent1]
        p2_list = [int(g) for g in parent2]

        offspring_list = [-1] * size

        # Choose segment bounds
        start, end = sorted(np.random.choice(size, 2, replace=False))

        # copy segment from p1 to offspring list
        segment_p1 = p1_list[start:end + 1]
        offspring_list[start:end + 1] = segment_p1

        # Genes in the P1 segment (used for conflict checking)
        copied_genes = set(segment_p1)

        # 2. Create mapping (P2 gene at index i -> P1 gene at index i)
        mapping = {
            p2_list[i]: p1_list[i]
            for i in range(start, end + 1)
        }

        #Fill remaining genes
        for i in range(size):
            if i in range(start, end + 1):
                continue  # Skip the segment

            gene_p2 = p2_list[i]

            # If the P2 gene is not in the segment copied from P1, copy it directly.
            if gene_p2 not in copied_genes:
                offspring_list[i] = gene_p2
            else:
                # If the P2 gene is in the P1 segment, resolve its mapping chain.
                # The map chain must lead to a gene that is not in the P1 segment.
                current_gene_to_map = gene_p2

                # Resolve the chain until the resulting gene is outside the mapping domain
                while current_gene_to_map in mapping:
                    current_gene_to_map = mapping[current_gene_to_map]

                offspring_list[i] = current_gene_to_map
        return np.array(offspring_list, dtype=int)

    def mutate(self, individual):
        mutated_individual = np.copy(individual)

        if self.num_cities < 2:
            return mutated_individual
        idx1, idx2 = np.random.choice(self.num_cities, 2, replace=False)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        segment_to_invert = mutated_individual[idx1:idx2 + 1]
        mutated_individual[idx1:idx2 + 1] = segment_to_invert[::-1]

        return mutated_individual

    def solution_distance(self, route1, route2):
        return calculate_solution_distance_tsp(route1, route2)


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
