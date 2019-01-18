import math
import numpy as np
import torch
from torch import nn


class GraphConv(nn.Module):
	"""
	Convolutional layer.
	Must specify the dimensions of the data for each layer.
	"""
	def __init__(self, emb_size):
		super().__init__() 
		self.emb_size = emb_size
		self.weight = nn.Parameter(torch.randn(emb_size, emb_size)) / math.sqrt(emb_size)

	def forward(self, data):
		out = torch.mm(torch.FloatTensor(data), self.weight)
		return out


class GraphPool(nn.Module):
	"""
	Pool layer that sums a node and its neighbors.
	"""
	def __init__(self):
		super().__init__()

	def _sum_neighbors(self, x, adj):
		updated = []
		for idx in range(adj.shape[0]):
			node_vec = np.array(x[idx].detach().numpy())
			n_idx = np.where(adj[idx] == 1)[0]
			neighbor_vecs = [x[n_idx[i]].detach().numpy() for i in range(len(n_idx))]
			neighbor_vecs.append(node_vec)
			neighbor_vecs = np.array(neighbor_vecs)
			s = np.sum(neighbor_vecs, axis=0)
			updated.append(s)
		return np.array(updated)

	def forward(self, x, adj):
		return self._sum_neighbors(x, adj)
