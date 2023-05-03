
import numpy as np
import networkx as nx

initial_state = 0


class ACO:

    def __init__(self, graph):
        self.distance = graph.matrix
        # tau
        self.pheromones = np.ones(self.distance.shape)
        # tau_delta
        self.pheromones_quantity = np.zeros(self.distance.shape)
        self.neta = 1/self.distance

        # Hyper params
        self.alpha = 1
        self.beta = 1
        # evaporation rate
        self.rho = 0.5
        self.max_iterations = 4
        self.ants = 2
        self.best_route = []

    def get_weight(self, unvisited):
        weight = self.pheromones**self.alpha * self.neta**self.beta
        unvisited_weight_matrix = weight[unvisited]

        return unvisited_weight_matrix

    def get_likelihood(self, unvisited_weight_matrix):
        likelihood = unvisited_weight_matrix/sum(unvisited_weight_matrix)
        return likelihood

    def choose_vertex(self, unvisited, likelihood, current_node):
        current_node = np.random.choice(
            unvisited, p=likelihood[:, current_node])

        return current_node

    def recalculate_pheromones_quantity(self, path, total_pheromones):
        for node in range(len(path)-1):
            self.pheromones_quantity[path[node],
                                     path[node + 1]] += total_pheromones

    def calculate_total_pheromones(self, total_distance):
        total_pheromones = 1/total_distance

        return total_pheromones

    def calculate_total_distance(self, path):
        total_distance = 0
        for node in range(len(path)-1):
            total_distance += self.distance[path[node], path[node + 1]]

        return total_distance

    def update_pheromones_matrix(self):
        self.pheromones = (1 - self.rho) * self.pheromones + \
            self.pheromones_quantity

    def get_best_route(self):
        self.best_route = list(self.pheromones.argmax(axis=1))
        print(f"Best solution found: {self.best_route}")

    def walk_path(self):
        for iteration in range(self.max_iterations):
            for ant in range(self.ants):
                path = []
                unvisited = [node for node in range(self.distance.shape[0])]
                current_node = initial_state
                unvisited.remove(current_node)
                path.append(current_node)

                while len(unvisited) != 0:
                    weight = self.get_weight(unvisited)
                    likelihood = self.get_likelihood(weight)
                    current_node = self.choose_vertex(
                        unvisited, likelihood, current_node)
                    unvisited.remove(current_node)
                    path.append(current_node)
                path.append(initial_state)
                total_distance = self.calculate_total_distance(path)

                total_pheromones = self.calculate_total_pheromones(
                    total_distance)
                self.recalculate_pheromones_quantity(path, total_pheromones)
                print(
                    f"Ant {ant} path: {path}. Total distance: {total_distance}. Total pheromones: {total_pheromones}")

        self.update_pheromones_matrix()
        return self.pheromones

    def solve(self):
        self.walk_path()
        self.get_best_route()
