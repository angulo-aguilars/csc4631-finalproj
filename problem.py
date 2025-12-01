import numpy as np
from abc import ABC, abstractmethod

class Problem(ABC):
    @abstractmethod
    def generate_random_solution(self):
        pass

    @abstractmethod
    def evaluate_fitness(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def calculate_solution_distance(self):
        pass

    def calculate_diversity(self):
        return


    class TSPProblem(Problem):
        def __init__(self):
            return

        def generate_random_solution(self):
            return

        def evaluate_fitness(self):
            return

        def crossover(self):
            return

        def mutate(self):
            return

        def calculate_solution_distance(self):
            return

