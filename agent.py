import numpy as np
from problem import Problem
from typing import List, Any, Tuple, Optional

class Agent:
    """
        A reusable GA agent with:
          - tournament selection
          - fitness sharing
          - adaptive mutation
          - steady-state replacement
    """

    def __init__(self,
                 problem,
                 population_size: int = 50,
                 max_generations: int = 200,
                 initial_mutation_rate: float = 0.1,
                 tournament_size: int = 5,
                 sharing_radius: float = 0.3,
                 replacement_count: int = 2,
                 rng_seed: Optional[int] = None,
                 maximize: bool = True):

        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations

        self.initial_mutation_rate = float(initial_mutation_rate)
        self.mutation_rate = float(initial_mutation_rate)

        self.tournament_size = int(tournament_size)
        self.sharing_radius = float(sharing_radius)
        self.replacement_count = int(replacement_count)

        self.maximize = bool(maximize)

        # seeded RNG
        self.rng = np.random.RandomState(rng_seed)

        # state
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

        raw_fitness = np.array([self.problem.evaluate_fitness(ind) for ind in self.population])

        for generation in range(self.max_generations):
            shared_fitness = self._fitness_sharing(raw_fitness)

            # track best fitness
            if self.maximize:
                best_raw = float(np.max(raw_fitness))
            else:
                best_raw = float(np.min(raw_fitness))
            self.best_fitness_history.append(best_raw)

            # measure diversity
            current_diversity = float(self.problem.calculate_diversity(self.population))
            self.diversity_history.append(current_diversity)

            #Adaptive mutation
            if generation > 5:
                recent_div = float(np.mean(self.diversity_history[-5:]))

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

                if self.rng.rand() < self.mutation_rate:
                    offspring = self.problem.mutate(offspring)
                new_offspring.append(offspring)

            offspring_fitnesses = np.array(
                [self.problem.evaluate_fitness(o) for o in new_offspring]
            )

            self._steady_state_replacement(new_offspring,
                                            offspring_fitnesses,
                                            raw_fitness)

            raw_fitness = np.array([self.problem.evaluate_fitness(ind)
                                    for ind in self.population])

        return self.population

    def _fitness_sharing(self, raw_fitness):
        shared_fitness = np.copy(raw_fitness)

        for i in range(len(self.population)):
            sharing_sum = 1.0  # always at least 1
            for j in range(len(self.population)):
                if i != j:
                    dist = self.problem.solution_distance(
                        self.population[i],
                        self.population[j]
                    )
                    if dist < self.sharing_radius:
                        sharing_sum += (1 - dist / self.sharing_radius)

            shared_fitness[i] = raw_fitness[i] / sharing_sum

        return shared_fitness

    def _tournament_selection(self, shared_fitness: np.ndarray):
        pop_n = len(self.population)
        k = min(self.tournament_size, pop_n)     # FIXED: ensures no crash
        k = max(k, 1)                            # FIXED: never 0

        # If population is too small: choose random single
        if k == 1:
            return self.population[self.rng.randint(pop_n)]

        indices = self.rng.choice(pop_n, size=k, replace=False)

        if self.maximize:
            best_idx = int(indices[np.argmax(shared_fitness[indices])])
        else:
            best_idx = int(indices[np.argmin(shared_fitness[indices])])

        return self.population[best_idx]

    def _steady_state_replacement(self,
                                  offspring: List[Any],
                                  offspring_fitnesses: np.ndarray,
                                  raw_fitness: np.ndarray) -> None:
        worst_indices = np.argsort(raw_fitness)[:self.replacement_count]
        for off, idx in zip(offspring, worst_indices):
            self.population[idx] = off

    def get_best_solution(self) -> Tuple[Any, float]:
        raw_fitness = np.array([self.problem.evaluate_fitness(ind) for ind in self.population])
        if self.maximize:
            best_idx = int(np.argmax(raw_fitness))
            best_val = float(np.max(raw_fitness))
        else:
            best_idx = int(np.argmin(raw_fitness))
            best_val = float(np.min(raw_fitness))
        return self.population[best_idx], best_val