import numpy as np
import networkx as nx

class Graph:

  def __init__(self, node_count):
    self.matrix = np.full((node_count, node_count), np.inf)
  
  def add_edge(self, first_index, second_index, distance, is_directed=False):
    self.matrix[first_index, second_index] = distance
    if not is_directed: self.matrix[second_index, first_index] = distance
  
  def to_networkx(self):
    graph_x = nx.Graph()
    for row in range(self.matrix.shape[0]):
      for column in range(self.matrix.shape[1]):
        if self.matrix[row, column] != np.inf:
          graph_x.add_edge(row, column, weight=self.matrix[row, column])
    position = nx.spring_layout(graph_x)
    labels = nx.get_edge_attributes(graph_x, 'weight')
    nx.draw_networkx(graph_x, pos=position, with_labels=True)
    nx.draw_networkx_edge_labels(graph_x, position, edge_labels=labels)