import math
import numpy as np
import torch
from torch import nn


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
	Pool layer that sums a node and its neighbors.
	"""
	def __init__(self, adj):
		super().__init__()
		self.adj = adj

	def _sum_neighbors(self, x):
		updated = []
		for idx in range(self.adj.shape[0]):
			node_vec = np.array(x[idx].detach().numpy())
			n_idx = np.where(self.adj[idx] == 1)[0]
			neighbor_vecs = [x[n_idx[i]].detach().numpy() for i in range(len(n_idx))]
			neighbor_vecs.append(node_vec)
			neighbor_vecs = np.array(neighbor_vecs)
			s = np.sum(neighbor_vecs, axis=0)
			updated.append(s)
		return np.array(updated)

	def forward(self, x):
		return self._sum_neighbors(x)
