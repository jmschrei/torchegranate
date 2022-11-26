# _base.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import torch

class Node(torch.nn.Module):
	def __init__(self, distribution, name):
		super().__init__()

		self.distribution = distribution
		self.name = name

class GraphMixin(torch.nn.Module):
	def add_node(self, node):
		self.nodes.append(node)

	def add_nodes(self, nodes):
		self.nodes.extend(nodes)

	def add_edge(self, start, end, probability):
		self.edges.append((start, end, probability))

