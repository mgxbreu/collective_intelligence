import math
import numpy as np
from app.constants import MAX_NUMBER
import random


class GeneticAlgorithmTSP:

    def __init__(self, population_size, distance_matrix, selection_rate=0.33, mutation_rate=0.16, iterations=50, population_count=6):
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.population_count = population_count
        self.selection_count = math.ceil(
            selection_rate * self.population_count)
        self.population = []
        self.fitness_list = {}

    def _initialize_population(self):
        for _ in range(self.population_count):
            random_population = np.random.permutation(
                self.population_size).tolist()
            self.population.append(random_population)

    def _fitness(self, path):
        total_distance = 0
        for node in range(len(path)-1):
            current_distance = self.distance_matrix[path[node], path[node + 1]]
            if current_distance == np.inf:
                return MAX_NUMBER
            total_distance += current_distance
        return total_distance

    def _generate_fitness_list(self):
        self.fitness_list = [self._fitness(
            chromosomes) for chromosomes in self.population]

    def _parent_selection(self):
        self._generate_fitness_list()
        indices = np.argsort(self.fitness_list)
        parents_selected = [self.population[i]
                            for i in indices[:self.selection_count]]
        return parents_selected

    def _crossover(self, parent1, parent2):
        half_parent_genes = int(len(parent1) / 2)
        child1 = parent1[:half_parent_genes] + parent2[half_parent_genes:]
        child2 = parent2[:half_parent_genes] + parent1[half_parent_genes:]

        section1 = parent1[:half_parent_genes]
        section2 = parent2[:half_parent_genes]

        mapped_genes1 = dict(zip(section1, section2))
        mapped_genes2 = dict(zip(section2, section1))

        child_to_be_born1 = section1 + \
            [gene if gene not in section1 else 'X' for gene in child1[half_parent_genes:]]
        child_to_be_born2 = section2 + \
            [gene if gene not in section2 else 'X' for gene in child2[half_parent_genes:]]

        for idx, gene in enumerate(child_to_be_born1):
            if gene == 'X':
                gene_mapped = child1[idx]

                while gene_mapped in child_to_be_born1:
                    gene_mapped = mapped_genes1[gene_mapped]

                child_to_be_born1[idx] = gene_mapped

        for idx, gene in enumerate(child_to_be_born2):
            if gene == 'X':
                gene_mapped = child2[idx]

                while gene_mapped in child_to_be_born2:
                    gene_mapped = mapped_genes2[gene_mapped]

                child_to_be_born2[idx] = gene_mapped

        child_to_be_born1 = self._mutate(child_to_be_born1)
        child_to_be_born2 = self._mutate(child_to_be_born2)

        return [child_to_be_born1, child_to_be_born2]

    def _mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            i = random.randint(0, len(chromosome)-1)
            j = random.randint(0, len(chromosome)-1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    def _find_best_path(self):
        best_path = min(self.population, key=self._fitness)
        best_path_distance = self._fitness(best_path)

        print(f"Current population: {self.population}")
        print(
            f"Best path found: {best_path}, with a length of {best_path_distance}")

    def _generate_next_generation(self, parents):
        next_generation = parents[:]
        while len(next_generation) < self.population_count:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            children = self._crossover(parent1, parent2)
            next_generation.extend(children)

        return next_generation

    def solve(self):
        self._initialize_population()
        for _ in range(self.iterations):
            parents = self._parent_selection()
            next_generation = self._generate_next_generation(parents)
            self.population = next_generation
            print(min(self.fitness_list))
        self._find_best_path()
