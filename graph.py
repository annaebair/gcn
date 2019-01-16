import csv

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

	def get_initial_vector(self, node, order, initial):
		if type(node) == Chem.rdchem.Atom: 
			idx = np.where(order == node.GetSymbol())
		elif type(node) == Chem.rdchem.Bond:
			idx = np.where(order == str(node.GetBondType()))
		return initial(torch.LongTensor(idx)).detach().numpy().flatten()

	def get_neighbor_sum(self, node, order, initial):
		if type(node) == Chem.rdchem.Atom:
			idx = node.GetIdx()
		elif type(node) == Chem.rdchem.Bond:
			idx = node.GetIdx() + self.num_atoms
		n_idx = np.where(self.adjacency_matrix[idx] == 1)
		neighbors = [self.nodes[n_idx[0][i]] for i in range(len(n_idx[0]))]
		n_vecs = []
		for n in neighbors:
			vec = self.get_initial_vector(n, order, initial)
			n_vecs.append(vec)
		return np.sum(np.array(n_vecs), axis=0)

	def create_fingerprint(self):
		order = np.array(list(self.node_set))
		initial = nn.Embedding(len(order), EMB_SIZE)
		for node in self.nodes:
			vec = self.get_initial_vector(node, order, initial)
			n_vecs = self.get_neighbor_sum(node, order, initial)
			self.fingerprint = np.sum([self.fingerprint, vec, n_vecs], axis=0)


def load_data():
	lst = []
	with open('data/sample_smiles.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			string = ''.join(row)
			if string != '-666' and 'smiles' not in string:
				lst.append(string)
	smiles = list(set(lst))
	vals = ['N' in i for i in smiles]
	df = pd.DataFrame({'smiles': smiles, 'vals': vals})
	df = df.sample(frac=1)
	x = list(df['smiles'])
	y = list(df['vals'])
	return x, y


def main():
	x, y = load_data()
	fingerprints = []
	for i in range(10):
		g = MolGraph(x[i])
		if g is not None:
			g.create_fingerprint()
			f = g.fingerprint
			fingerprints.append(f)
	fingerprints = np.array(fingerprints)
	print(fingerprints)


if __name__ == "__main__":
	main()
