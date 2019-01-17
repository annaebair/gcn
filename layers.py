import math
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F

from rdkit import Chem

from adj_rep import AdjacencyRep
from graph_rep import MolGraph


class GraphConv(nn.Module):
	"""
	Convolutional layer.
	Must specify the dimensions of the data for each layer.
	"""
	def __init__(self, feat_size, out_dim):
		super().__init__() 
		self.feat_size = feat_size
		self.out_dim = out_dim
		self.weight = nn.Parameter(torch.randn(feat_size, out_dim)) / math.sqrt(feat_size)

	def forward(self, data):
		# data must be n x feat_size matrix
		out = torch.mm(torch.FloatTensor(data), self.weight)
		return out


class GraphPool(nn.Module):
	"""
	Pool layer that combines a node and its neighbors.
	"""
	def __init__(self, graph):
		super().__init__()
		self.graph = graph

	def forward(self, x):
		vecs = []
		for node in self.graph.nodes:
			node_vec = self.graph.get_initial_vector(node) 
			neighbors = self.graph.get_neighbor_sum(node)
			vec_sum = np.sum([node_vec, neighbors], axis=0)
			vecs.append(vec_sum)
		return np.array(vecs)
