import math
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F

from rdkit import Chem

from graph_rep import MolGraph
from layers import GraphConv, GraphPool


class GCN(nn.Module):
	def __init__(self, feat_size_1, hidden, out_dim_2, graph):
		super().__init__()
		self.adj = graph.adjacency_matrix
		self.gc1 = GraphConv(feat_size_1, hidden)
		self.gc2 = GraphConv(hidden, out_dim_2)
		self.pool1 = GraphPool(self.adj)
		self.pool2 = GraphPool(self.adj)

	def forward(self, x):
		x = self.gc1(x)
		x = self.pool1(x)
		x = self.gc2(x)
		x = self.pool2(x)
		return x
