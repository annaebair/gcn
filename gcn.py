import math
import numpy as np
import torch
from torch import nn

from layers import GraphConv, GraphPool


class GCN(nn.Module):
	def __init__(self, adj_mat, emb_size):
		super().__init__()
		self.adj = adj_mat
		self.gc1 = GraphConv(emb_size, emb_size)
		self.gc2 = GraphConv(emb_size, emb_size)
		self.gc3 = GraphConv(emb_size, emb_size)
		self.pool1 = GraphPool(self.adj)
		self.pool2 = GraphPool(self.adj)
		self.pool3 = GraphPool(self.adj)

	def forward(self, x):
		x = self.gc1(x)
		x = self.pool1(x)
		x = self.gc2(x)
		x = self.pool2(x)
		x = self.gc3(x)
		x = self.pool3(x)
		return x
