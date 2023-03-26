# _base.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import torch


class GraphMixin(torch.nn.Module):
	def add_node(self, node):
		self.nodes.append(node)

	def add_nodes(self, nodes):
		self.nodes.extend(nodes)

	def add_edge(self, start, end, probability=None):
		if probability is None:
			edge = start, end
		else:
			edge = start, end, probability

		self.edges.append(edge)
