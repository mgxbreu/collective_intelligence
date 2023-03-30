from graph import Graph
import numpy as np
from constants import MAX_NUMBER
import math

nodes = 4
cities = Graph(nodes)
cities.add_edge(0,1,3)
cities.add_edge(0,3,5)
cities.add_edge(1,0,2)
cities.add_edge(1,3,4)
cities.add_edge(2,1,1)
cities.add_edge(3,2,2)

distance = cities.matrix

print(cities.matrix)

class GeneticAlgorithmTSP():

    def __init__(self, selection_rate=0.66, mutation_rate= 0.16, iterations= 4, population_size= 4):
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.population_size = population_size
        self.population_length= nodes
        self.selection_count = math.ceil(selection_rate * self.population_size)
        self.population = []  
        self.fitness_dict = {}  


    def _initialize_population(self):
        population = []
        for i in range(self.population_length):
            random_population =  np.random.permutation(self.population_length).tolist()
            population.append(random_population)

        return population

    def _fitness(self, path):
        total_distance = 0
        for node in range(len(path)-1):
            current_distance = distance[path[node], path[node + 1]]
            if current_distance == np.inf:
                return MAX_NUMBER
            total_distance += current_distance
        #check why
        # total_distance += distance[path[-1], path[0]]
        return total_distance

    def _generate_fitness_dict(self):
        self.fitness_dict = {idx: self._fitness(parent) for idx, parent in enumerate(self.population)}
        # print(self.fitness_dict)
    #error on key
    def _parent_selection(self):
        self._generate_fitness_list()
        indices = np.argsort(self.fitness_list)
        # check why
        parents_selected = [self.population[i] for i in indices[:self.selection_count]]
        return parents_selected

    
    def _crossover(self, parent1, parent2):
        half_parent_genes = int(len(parent1) / 2)
        child1 = parent1[:half_parent_genes] + parent2[half_parent_genes:]
        child2 = parent2[:half_parent_genes] + parent1[half_parent_genes:]

        section1 = parent1[:half_parent_genes]
        section2 = parent2[:half_parent_genes]

        mapped_genes1 = dict(zip(section1, section2))
        mapped_genes2 = dict(zip(section2, section1))

        child_to_be_born1 = section1 + [gene if gene not in section1 else 'X' for gene in child1[half_parent_genes:]]
        child_to_be_born2 = section2 + [gene if gene not in section2 else 'X' for gene in child2[half_parent_genes:]]
        # print(child_to_be_born1)

        for idx, gene in enumerate(child_to_be_born1):
            if gene == 'X':
                # print(mapped_genes1[1])
                gene_mapped = child1[idx]

                while gene_mapped in child_to_be_born1:
                    gene_mapped = mapped_genes1[gene_mapped]

                child_to_be_born1[idx] = gene_mapped

        for idx, gene in enumerate(child_to_be_born2):
            if gene == 'X':
                # print(mapped_genes1[1])
                gene_mapped = child2[idx]

                while gene_mapped in child_to_be_born2:
                    gene_mapped = mapped_genes2[gene_mapped]

                child_to_be_born2[idx] = gene_mapped

        return [child_to_be_born1, child_to_be_born2]

    def _find_best_path(self):
        best_path_index =  min(self.fitness_dict, key=self.fitness_dict.get)
        best_path_length = self.fitness_dict[best_path_index]
        best_path = self.population[best_path_index]

        print("best path")
        print(f"Best path found: {best_path}, with a length of {best_path_length}") 
        
    def solve(self):
        self.population = self._initialize_population()
        parents = self._parent_selection()
        print(parents)
        # parent1 = parents[0]
        # parent2 = parents[1]
        # children = self._crossover(parent1, parent2)
        # print(children)
        # for child in children:
        #     if child in parents:
        #         children.remove(child)
        # print(self.fitness_list)
        # self._generate_fitness_list()
        # self._find_best_path()
        
        
TSP = GeneticAlgorithmTSP()
TSP.solve()

#TODO: mutacion
#TODO: fix errors
#TODO: make class receive params