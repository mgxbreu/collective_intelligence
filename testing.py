from graph import Graph
import numpy as np
from constants import MAX_NUMBER
import math
import random
from genetic_algorithm import GeneticAlgorithmTSP
 
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

TSP = GeneticAlgorithmTSP(nodes, distance)
TSP.solve()