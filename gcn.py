import math
import numpy as np
import torch
from torch import nn

from layers import GraphConv, GraphPool


class GCN(nn.Module):
	def __init__(self, emb_size):
		super().__init__()
		self.gc1 = GraphConv(emb_size)
		self.gc2 = GraphConv(emb_size)
		self.gc3 = GraphConv(emb_size)
		self.pool1 = GraphPool()
		self.pool2 = GraphPool()
		self.pool3 = GraphPool()

	def forward(self, x, adj):
		x = self.gc1(x)
		x = self.pool1(x, adj)
		x = self.gc2(x)
		x = self.pool2(x, adj)
		x = self.gc3(x)
		x = self.pool3(x, adj)
		return x


class GCN_Drug_Embedder(nn.Module):
	"""

	"""
	def __init__(self, vocab_size, emb_size):
		super().__init__()
		self.vocab_size = vocab_size
		self.embed = nn.Embedding(vocab_size, emb_size)

		self.gc1 = GraphConv(emb_size)
		self.gc2 = GraphConv(emb_size)
		self.gc3 = GraphConv(emb_size)
		self.pool1 = GraphPool()
		self.pool2 = GraphPool()
		self.pool3 = GraphPool()

	def forward(self, indices, adj_mat):
		# TODO: add activations/make this non trivial
		x = self.embed(torch.LongTensor(indices))
		x = self.gc1(x)
		x = self.pool1(x, adj_mat)
		x = self.gc2(x)
		x = self.pool1(x, adj_mat)
		return np.sum(x, axis=0)
