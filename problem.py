import numpy as np
from abc import ABC, abstractmethod
from helper_functions import *

from bokeh.sampledata.gapminder import population


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

    @abstractmethod
    def calculate_solution_distance(self, individual1, individual2):
        pass

    def calculate_diversity(self):
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distances.append(self.calculate_solution_distance(population[i], population[j]))

            return np.mean(distances) if distances else 0


class TSPProblem(Problem):
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = calculate_distance_matrix(cities)

    def generate_random_solution(self):
        return np.random.permutation(self.num_cities)

    def evaluate_fitness(self, individual):
        distance = 0
        for i in range(self.num_cities):
            initial_city = individual[i]
            last_city = individual[(i + 1) % self.num_cities]
            distance += self.distance_matrix[initial_city][last_city]
        return 1.0 / distance


    def crossover(self, parent1, parent2):
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

    def calculate_solution_distance(self, individual1, individual2):
        return calculate_solution_distance_tsp(individual1, individual2)

class GCPProblem(Problem):
