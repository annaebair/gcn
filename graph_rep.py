import numpy as np
import torch
from torch import nn

from rdkit import Chem


class MolGraph(object):
	"""
	Graph representation of SMILES string.
	Loosely based on https://github.com/HIPS/neural-fingerprint
	"""

	def __init__(self, smiles, emb_size):
		self.mol = Chem.MolFromSmiles(smiles)
		self.emb_size = emb_size

		self.atoms_list = self.mol.GetAtoms()
		self.bonds_list = self.mol.GetBonds()
		self.nodes = list(self.atoms_list) + list(self.bonds_list)
		self.node_set = self._node_set()
		self.num_nodes = len(self.nodes)
		self.adjacency_matrix = self._adjacency_matrix()
		self.order = np.array(list(self.node_set))
		self.initial = nn.Embedding(len(self.order), self.emb_size)

	def _adjacency_matrix(self):
		adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
		for bond in self.mol.GetBonds():
			begin_idx = bond.GetBeginAtom().GetIdx()
			end_idx = bond.GetEndAtom().GetIdx()
			bond_idx = bond.GetIdx() + len(self.mol.GetAtoms()) 
			adjacency_matrix[begin_idx, bond_idx] = 1
			adjacency_matrix[bond_idx, begin_idx] = 1
			adjacency_matrix[end_idx, bond_idx] = 1
			adjacency_matrix[bond_idx, end_idx] = 1
		return adjacency_matrix

	def _node_set(self):
		node_set = set()
		for atom in self.atoms_list:
			sym = atom.GetSymbol()
			node_set.add(sym)
		for bond in self.bonds_list:
			btype = bond.GetBondType()
			node_set.add(btype)
		return node_set

	def _get_initial_vector(self, node):
		if type(node) == Chem.rdchem.Atom: 
			idx = np.where(self.order == node.GetSymbol())
		elif type(node) == Chem.rdchem.Bond:
			idx = np.where(self.order == str(node.GetBondType()))
		return self.initial(torch.LongTensor(idx)).detach().numpy().flatten()

	def get_initial(self):
		mat = []
		for node in self.nodes:
			vec = self._get_initial_vector(node)
			mat.append(vec)
		return np.array(mat)

	def get_adj_mat(self):
		return self.adjacency_matrix
