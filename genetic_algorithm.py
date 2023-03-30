import numpy as np
import networkx as nx
from graph import Graph
import random


# Number of vertices
nV = 4
INF = 999

# # Algorithm 
# def floyd(G):
#     dist = list(map(lambda p: list(map(lambda q: q, p)), G))

#     # Adding vertices individually
#     for r in range(nV):
#         for p in range(nV):
#             for q in range(nV):
#                 # print(dist[r])
#                 # print(dist[p][q])
#                 # print(dist[p][r])
#                 # if  q ==  r:
#                 #   dist[p][q] == 0
#                 # else: 
#                 print
#                 dist[p][q] = min(dist[p][q], dist[p][r] + dist[r][q])
#     sol(dist)

# # Printing the output
# def sol(dist):
#     for p in range(nV):
#         for q in range(nV):
#             if(dist[p][q] == INF):
#                 print("INF", end=" ")
#             else:
#                 print(dist[p][q], end="  ")
#         print(" ")

# G = [[0, 5, INF, INF],
#          [50, 0, 15, 5],
#          [30, INF, 0, 15],
#          [15, INF, 5, 0]]
# # floyd(G)

# graph.matrix

# floyd(graph.matrix)

# initial_state = 0

# class ACO:

#   def __init__(self, graph):
#     self.distance= graph.matrix
#     #tau
#     self.pheromones = np.ones(self.distance.shape)
#     #tau_delta
#     self.pheromones_quantity = np.zeros(self.distance.shape)
#     self.neta = 1/self.distance

#     ## Hyper params
#     self.alpha = 1
#     self.beta = 1
#     #evaporation rate
#     self.rho = 0.5
#     self.max_iterations = 4
#     self.ants = 2
#     self.best_route = []

#   def get_weight(self, unvisited):
#     weight = self.pheromones**self.alpha * self.neta**self.beta
#     unvisited_weight_matrix = weight[unvisited]

#     return unvisited_weight_matrix

#   def get_likelihood(self, unvisited_weight_matrix):
#     likelihood = unvisited_weight_matrix/sum(unvisited_weight_matrix)

#     return likelihood

#   def choose_vertex(self, unvisited, likelihood, current_node):
#     current_node = np.random.choice(unvisited, p=likelihood[:, current_node])
    
#     return current_node
  
#   def recalculate_pheromones_quantity(self, path, total_pheromones):
#     for node in range(len(path)-1):
#           self.pheromones_quantity[path[node], path[node + 1]] += total_pheromones

#   def calculate_total_pheromones(self, total_distance):
#       total_pheromones = 1/total_distance

#       return total_pheromones

#   def calculate_total_distance(self, path):
#     total_distance = 0
#     for node in range(len(path)-1):
#       total_distance += self.distance[path[node], path[node + 1]]

#     return total_distance

#   def update_pheromones_matrix(self):
#     self.pheromones = (1 - self.rho) * self.pheromones + self.pheromones_quantity

#   def get_best_route(self):
#     self.best_route = list(self.pheromones.argmax(axis=1))
    
#   def find_path(self):
#     path = []
#     unvisited = [node for node in range(self.distance.shape[0])]
#     current_node = initial_state
#     unvisited.remove(current_node)
#     path.append(current_node)

#     while len(unvisited) != 0:
#       weight = self.get_weight(unvisited)
#       likelihood = self.get_likelihood(weight)
#       current_node = self.choose_vertex(unvisited, likelihood, current_node)
#       unvisited.remove(current_node)
#       path.append(current_node)
#     path.append(initial_state)

#     return path

#   def walk_path(self):
#     for iteration in range(self.max_iterations):
#       for ant in range(self.ants):
#         path = self.find_path()
#         total_distance = self.calculate_total_distance(path)
#         total_pheromones = self.calculate_total_pheromones(total_distance)
#         self.recalculate_pheromones_quantity(path, total_pheromones)

      
#       self.update_pheromones_matrix()


#   def start(self):
#     self.walk_path()
#     self.get_best_route()

# ants_algorithm = ACO(graph)
# best_route = ants_algorithm.start()
# print(best_route)

# import numpy as np
# import random

# class GeneticAlgorithmTSP:
#     def __init__(self, distances, population_size=100, elite_size=20, mutation_rate=0.01, generations=500):
#         self.distances = distances
#         self.population_size = population_size
#         self.elite_size = elite_size
#         self.mutation_rate = mutation_rate
#         self.generations = generations

#     def generate_initial_population(self):
#         population = []
#         for i in range(self.population_size):
#             chromosome = list(range(self.distances.shape[0]))
#             random.shuffle(chromosome)
#             population.append(chromosome)
#         return population

#     def evaluate_fitness(self, chromosome):
#         fitness = 0
#         for i in range(len(chromosome)-1):
#             print(chromosome)
#             fitness += self.distances[chromosome[i], chromosome[i+1]]
#         fitness += self.distances[chromosome[-1], chromosome[0]]
#         return fitness

#     def select_parents(self, population):
#         fitnesses = [self.evaluate_fitness(chromosome) for chromosome in population]
#         indices = np.argsort(fitnesses)
#         parents = [population[i] for i in indices[:self.elite_size]]
#         return parents

#     def crossover(self, parent1, parent2):
#         child = [-1 for _ in range(len(parent1))]
#         start = random.randint(0, len(parent1)-1)
#         end = random.randint(start, len(parent1)-1)
#         child[start:end+1] = parent1[start:end+1]
#         for i in range(len(parent2)):
#             if parent2[i] not in child:
#                 for j in range(len(child)):
#                     if child[j] == -1:
#                         child[j] = parent2[i]
#                         break
#         return child

#     def mutate(self, chromosome):
#         if random.random() < self.mutation_rate:
#             i = random.randint(0, len(chromosome)-1)
#             j = random.randint(0, len(chromosome)-1)
#             chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
#         return chromosome

#     def generate_next_population(self, parents):
#         next_population = parents[:]
#         while len(next_population) < self.population_size:
#             parent1 = random.choice(parents)
#             parent2 = random.choice(parents)
#             child = self.crossover(parent1, parent2)
#             child = self.mutate(child)
#             next_population.append(child)
#         return next_population

#     def solve(self):
#         population = self.generate_initial_population()
#         for generation in range(self.generations):
#             parents = self.select_parents(population)
#             next_population = self.generate_next_population(parents)
#             population = next_population
#         best_chromosome = min(population, key=self.evaluate_fitness)
#         best_fitness = self.evaluate_fitness(best_chromosome)
#         return (best_chromosome, best_fitness)

# """

# The `GeneticAlgorithmTSP` class takes as input a distance matrix, which represents the distances between cities, and several parameters that affect the behavior of the genetic algorithm. The `generate_initial_population()` method creates an initial population of chromosomes, where each chromosome is a random permutation of the cities. The `evaluate_fitness()` method computes the fitness of a chromosome, which in this case is the length of the path it represents. The `select_parents()` method selects the best chromosomes to be parents of the next generation, based on their fitness. The `crossover()` method combines two chromosomes to create a new one, and the `mutate()` method introduces random changes to a chromosome. Finally, the `generate_next_population()` method creates the next generation of chromosomes by applying the genetic operators to the selected parents.

# The `solve()` method runs the genetic algorithm for a fixed number of generations and returns the best chromosome found, along with its fitness. To use the class, you can create an instance of it with the desired parameters, and then call the `solve()` method:

# """
# distances = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
# ga = GeneticAlgorithmTSP(distances)
# best_chromosome, best_fitness = ga.solve()
# print(best_chromosome)
# print(best_fitness)

"""

This example code creates a `GeneticAlgorithmTSP` object with a distance matrix of size 3x3, where the distance from city i to city j is given by the (i,j)-th entry of the matrix. The `solve()` method is then called to find the best path and its length. Note that the distance matrix should be symmetric and have zeros on the diagonal.

I hope this helps! Let me know if you have any questions.Sure, here is an implementation of a Python class that solves the Traveling Salesman Problem using a genetic algorithm:

"""
class GeneticAlgorithmTSP:
    # def __init__(self, distances, population_size=100, elite_size=20, mutation_rate=0.01, generations=500):
    def __init__(self, distances, population_size=5, elite_size=20, mutation_rate=0.01, generations=3):
        self.distances = distances
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            chromosome = list(range(self.distances.shape[0]))
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def evaluate_fitness(self, chromosome):
        fitness = 0
        for i in range(len(chromosome)-1):
            fitness += self.distances[chromosome[i], chromosome[i+1]]
        fitness += self.distances[chromosome[-1], chromosome[0]]
        return fitness

    def select_parents(self, population):
        fitnesses = [self.evaluate_fitness(chromosome) for chromosome in population]
        indices = np.argsort(fitnesses)
        parents = [population[i] for i in indices[:self.elite_size]]
        return parents

    def crossover(self, parent1, parent2):
        child = [-1 for _ in range(len(parent1))]
        start = random.randint(0, len(parent1)-1)
        end = random.randint(start, len(parent1)-1)
        child[start:end+1] = parent1[start:end+1]
        for i in range(len(parent2)):
            if parent2[i] not in child:
                for j in range(len(child)):
                    if child[j] == -1:
                        child[j] = parent2[i]
                        break
        return child

    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            i = random.randint(0, len(chromosome)-1)
            j = random.randint(0, len(chromosome)-1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    def generate_next_population(self, parents):
        next_population = parents[:]
        while len(next_population) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            next_population.append(child)
        return next_population

    def solve(self):
        population = self.generate_initial_population()
        for generation in range(self.generations):
            parents = self.select_parents(population)
            next_population = self.generate_next_population(parents)
            population = next_population
        best_chromosome = min(population, key=self.evaluate_fitness)
        best_fitness = self.evaluate_fitness(best_chromosome)
        return (best_chromosome, best_fitness)

"""

The `GeneticAlgorithmTSP` class takes as input a distance matrix, which represents the distances between cities, and several parameters that affect the behavior of the genetic algorithm. The `generate_initial_population()` method creates an initial population of chromosomes, where each chromosome is a random permutation of the cities. The `evaluate_fitness()` method computes the fitness of a chromosome, which in this case is the length of the path it represents. The `select_parents()` method selects the best chromosomes to be parents of the next generation, based on their fitness. The `crossover()` method combines two chromosomes to create a new one, and the `mutate()` method introduces random changes to a chromosome. Finally, the `generate_next_population()` method creates the next generation of chromosomes by applying the genetic operators to the selected parents.

The `solve()` method runs the genetic algorithm for a fixed number of generations and returns the best chromosome found, along with its fitness. To use the class, you can create an instance of it with the desired parameters, and then call the `solve()` method:
"""
