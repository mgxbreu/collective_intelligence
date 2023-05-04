from app.algorithms.genetic_algorithm import GeneticAlgorithmTSP
from app.classes.graph import Graph
import sys
import numpy as np
import unittest


class TestGeneticAlgorithmTSP(unittest.TestCase):

    def setUp(self):
        self.node_count = 5
        self.graph = Graph(self.node_count)
        self.graph.add_edge(0, 1, 2)
        self.graph.add_edge(0, 2, 5)
        self.graph.add_edge(0, 3, 1)
        self.graph.add_edge(0, 4, 6)
        self.graph.add_edge(1, 2, 3)
        self.graph.add_edge(1, 3, 2)
        self.graph.add_edge(1, 4, 5)
        self.graph.add_edge(2, 3, 4)
        self.graph.add_edge(2, 4, 3)
        self.graph.add_edge(3, 4, 5)

    def test_initialize_population(self):
        algorithm = GeneticAlgorithmTSP(population_size=self.node_count, distance_matrix=self.graph.matrix,
                                        population_count=3, iterations=10)
        algorithm._initialize_population()
        self.assertEqual(len(algorithm.population), 3)
        self.assertTrue(
            all(len(chromosome) == self.node_count for chromosome in algorithm.population))

    # def test_fitness(self):
    #     # Test if the fitness function is calculated correctly
    #     algorithm = GeneticAlgorithmTSP(population_size=self.node_count, distance_matrix=self.graph.matrix,
    #                                     population_count=3, iterations=10)
    #     path = [0, 2, 4, 3, 1]
    #     fitness = algorithm._fitness(path)
    #     self.assertEqual(fitness, 17)

    def test_parent_selection(self):
        algorithm = GeneticAlgorithmTSP(population_size=self.node_count, distance_matrix=self.graph.matrix,
                                        population_count=6, iterations=10)
        algorithm._initialize_population()
        parents_selected = algorithm._parent_selection()
        self.assertEqual(len(parents_selected), 2)

    def test_crossover(self):
        algorithm = GeneticAlgorithmTSP(population_size=self.node_count, distance_matrix=self.graph.matrix,
                                        population_count=6, iterations=10)
        parent1 = [0, 2, 4, 3, 1]
        parent2 = [1, 4, 2, 3, 0]
        children = algorithm._crossover(parent1, parent2)
        self.assertTrue(
            all(len(child) == self.node_count for child in children))
        self.assertTrue(all(set(child) == set(parent1) for child in children) or
                        all(set(child) == set(parent2) for child in children))

    def test_mutate(self):
        algorithm = GeneticAlgorithmTSP(population_size=self.node_count, distance_matrix=self.graph.matrix,
                                        population_count=6, iterations=10)
        chromosome = [0, 2, 4, 3, 1]
        mutated_chromosome = algorithm._mutate(chromosome)
        self.assertEqual(len(chromosome), len(mutated_chromosome))
        self.assertTrue(set(chromosome) == set(mutated_chromosome))

    def test_find_best_path(self):
        pass

    def test_generate_next_generation(self):
        pass

    def test_solve(self):
        pass


if __name__ == '__main__':
    unittest.main()
