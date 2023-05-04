# create a test for ACO class

import unittest
from ACO import ACO
from Graph import Graph
import numpy as np
from constants import MAX_NUMBER
import math
import random


class TestACO(unittest.TestCase):

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

    def test_initialize_pheromone_matrix(self):
        pass
