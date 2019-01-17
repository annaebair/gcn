import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F

from rdkit import Chem

EMB_SIZE = 5


class MolGraph(object):
	"""
	Graph representation of SMILES string.
	Loosely based on https://github.com/HIPS/neural-fingerprint
	"""

	def __init__(self, smiles):
		self.mol = Chem.MolFromSmiles(smiles)
		self.atoms_list = self.mol.GetAtoms()
		self.bonds_list = self.mol.GetBonds()
		self.nodes = list(self.atoms_list) + list(self.bonds_list)
		self.node_set = set([])
		self.num_atoms = len(self.atoms_list)
		self.num_bonds = len(self.bonds_list)
		self.num_nodes = self.num_atoms + self.num_bonds
		self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
		self.processing()
		self.fingerprint = np.zeros(5)
		self.order = np.array(list(self.node_set))
		self.initial = nn.Embedding(len(self.order), EMB_SIZE)

	def processing(self):
		for atom in self.atoms_list:
			sym = atom.GetSymbol()
			idx = atom.GetIdx()
			self.node_set.add(sym)

		for bond in self.bonds_list:
			begin_idx = bond.GetBeginAtom().GetIdx()
			end_idx = bond.GetEndAtom().GetIdx()
			bond_idx = bond.GetIdx() + self.num_atoms 
			btype = bond.GetBondType()
			self.node_set.add(btype)

			self.adjacency_matrix[begin_idx, bond_idx] = 1
			self.adjacency_matrix[bond_idx, begin_idx] = 1
			self.adjacency_matrix[end_idx, bond_idx] = 1
			self.adjacency_matrix[bond_idx, end_idx] = 1

	def get_initial_vector(self, node):
		if type(node) == Chem.rdchem.Atom: 
			idx = np.where(self.order == node.GetSymbol())
		elif type(node) == Chem.rdchem.Bond:
			idx = np.where(self.order == str(node.GetBondType()))
		return self.initial(torch.LongTensor(idx)).detach().numpy().flatten()

	def get_neighbor_sum(self, node):
		if type(node) == Chem.rdchem.Atom:
			idx = node.GetIdx()
		elif type(node) == Chem.rdchem.Bond:
			idx = node.GetIdx() + self.num_atoms
		n_idx = np.where(self.adjacency_matrix[idx] == 1)
		neighbors = [self.nodes[n_idx[0][i]] for i in range(len(n_idx[0]))]
		n_vecs = []
		for n in neighbors:
			vec = self.get_initial_vector(n)
			n_vecs.append(vec)
		return np.sum(np.array(n_vecs), axis=0)

	def sigmoid(self, x):
		return 0.5*(np.tanh(x) + 1)

	def create_fingerprint(self):
		for node in self.nodes:
			vec = self.get_initial_vector(node)
			n_vecs = self.get_neighbor_sum(node)
			vec_sum = np.sum([vec, n_vecs], axis=0)
			sig = self.sigmoid(vec_sum)
			self.fingerprint = np.sum([self.fingerprint, sig], axis=0)

	def get_initial(self):
		mat = []
		for node in self.nodes:
			vec = self.get_initial_vector(node)
			mat.append(vec)
		return np.array(mat)
