from classes.graph import Graph
import numpy as np
from constants import MAX_NUMBER
import math
import random
from algorithms.genetic_algorithm import GeneticAlgorithmTSP

nodes = 6
cities = Graph(nodes)
cities.add_edge(0, 1, 3)
cities.add_edge(0, 2, 4)
cities.add_edge(0, 3, 5)
cities.add_edge(0, 4, 6)
cities.add_edge(0, 5, 7)
cities.add_edge(1, 2, 8)
cities.add_edge(1, 3, 9)
cities.add_edge(1, 4, 10)
cities.add_edge(1, 5, 11)
cities.add_edge(2, 3, 12)
cities.add_edge(2, 4, 13)
cities.add_edge(2, 5, 14)
cities.add_edge(3, 4, 15)
cities.add_edge(3, 5, 16)
cities.add_edge(4, 5, 17)


distance = cities.matrix

print(cities.matrix)

TSP = GeneticAlgorithmTSP(nodes, distance)
TSP.solve()
