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
			n_idx = np.where(adj[idx] == 1)[0]
			neighbor_vecs = [x[n_idx[i]] for i in range(len(n_idx))]
			all_vecs = neighbor_vecs + [x[idx]]
			c = torch.stack(all_vecs, 0)
			s = torch.sum(c, 0)
			updated.append(s)
		return torch.stack(updated, 0)

	def forward(self, x, adj):
		return self._sum_neighbors(x, adj)
