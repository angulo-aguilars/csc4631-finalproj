import numpy as np
from problem import Problem
from typing import List, Any, Tuple

class Agent:
    def __init__(self, problem: Problem, population_size=50, max_generations=200,
                 initial_mutation_rate=0.1, tournament_size=5,
                 sharing_radius=0.1, replacement_count=2):

        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations

        self.initial_mutation_rate = initial_mutation_rate
        self.mutation_rate = initial_mutation_rate

        self.tournament_size = tournament_size
        self.sharing_radius = sharing_radius
        self.replacement_count = replacement_count

        self.population: List[Any] = []
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    def initialize_population(self):
        self.population = [
            self.problem.generate_random_solution()
            for _ in range(self.population_size)
        ]

    def run_evolution(self):
        self.initialize_population()

        for generation in range(self.max_generations):
            print(f"[GA] Generation {generation} ...", end="\r")
            raw_fitness = np.array([
                self.problem.evaluate_fitness(ind)
                for ind in self.population
            ])

            shared_fitness = self._fitness_sharing(raw_fitness)

            best_raw_fitness = float(np.max(raw_fitness))
            self.best_fitness_history.append(best_raw_fitness)

            current_diversity = self.problem.calculate_diversity(self.population)
            self.diversity_history.append(current_diversity)

            #Adaptive mutation
            if generation > 5:
                recent_div = np.mean(self.diversity_history[-10:])

                if current_diversity < recent_div:
                    self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
                else:
                    self.mutation_rate = max(
                        self.initial_mutation_rate,
                        self.mutation_rate * 0.95
                    )

            new_offspring = []

            for _ in range(self.replacement_count):
                parent1 = self._tournament_selection(shared_fitness)
                parent2 = self._tournament_selection(shared_fitness)

                offspring = self.problem.crossover(parent1, parent2)

                if np.random.rand() < self.mutation_rate:
                    offspring = self.problem.mutate(offspring)

                new_offspring.append(offspring)

            self._steady_state_replacement(new_offspring, raw_fitness)

        return self.population

    def _fitness_sharing(self, raw_fitness):
        shared_fitness = np.copy(raw_fitness)
        n = len(self.population)

        SAMPLE_SIZE_K = min(10, n - 1)

        for i in range(n):
            neighbor_indices = np.random.choice(
                [j for j in range(n) if j != i],
                SAMPLE_SIZE_K,
                replace=False
            )

            sharing_factor_sum = 0

            for j in neighbor_indices:
                dist = self.problem.solution_distance(
                    self.population[i],
                    self.population[j]
                )

                if dist < self.sharing_radius:
                    sharing_factor_sum += (1 - dist / self.sharing_radius)
            shared_fitness[i] /= (1 + sharing_factor_sum)

        return shared_fitness

    def _tournament_selection(self, shared_fitness):
        indices = np.random.choice(
            len(self.population),
            self.tournament_size,
            replace=False
        )
        best_idx = indices[np.argmax([shared_fitness[i] for i in indices])]
        return self.population[best_idx]

    def _steady_state_replacement(self, offspring: List[Any], raw_fitness: np.ndarray) -> None:
        """Replace the worst individuals in the population with new offspring."""
        worst_indices = np.argsort(raw_fitness)[:self.replacement_count]
        for off, idx in zip(offspring, worst_indices):
            self.population[idx] = off

    def get_best_solution(self) -> Tuple[Any, float]:
        """Return the best individual and its raw fitness."""
        raw_fitness = [
            self.problem.evaluate_fitness(ind)
            for ind in self.population
        ]
        best_idx = int(np.argmax(raw_fitness))
        return self.population[best_idx], raw_fitness[best_idx]