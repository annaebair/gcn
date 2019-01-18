import math
import csv
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from rdkit import Chem

from graph_rep import MolGraph
from gcn import GCN, GCN_Drug_Embedder

EMB_SIZE = 5

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

def adjacency_matrix(mol):
	num_nodes = len(list(mol.GetAtoms())) + len(list(mol.GetBonds()))
	adjacency_matrix = np.zeros((num_nodes, num_nodes))
	for bond in mol.GetBonds():
		begin_idx = bond.GetBeginAtom().GetIdx()
		end_idx = bond.GetEndAtom().GetIdx()
		bond_idx = bond.GetIdx() + len(mol.GetAtoms()) 
		adjacency_matrix[begin_idx, bond_idx] = 1
		adjacency_matrix[bond_idx, begin_idx] = 1
		adjacency_matrix[end_idx, bond_idx] = 1
		adjacency_matrix[bond_idx, end_idx] = 1
	return adjacency_matrix

def create_sym_to_idx(smiles):
	nodes_set = set()
	for s in smiles:
		mol = Chem.MolFromSmiles(s)
		atoms = mol.GetAtoms()
		bonds = mol.GetBonds()
		nodes_set.update([a.GetSymbol() for a in atoms])
		nodes_set.update([b.GetBondType() for b in bonds])
	nodes_list = list(nodes_set)
	sym_to_idx = {sym: idx for idx, sym in enumerate(nodes_list)}
	return sym_to_idx

def preprocess():
	# This is what should get merged into MultitaskModelBase
	smiles, _ = load_data()
	sym_to_idx = create_sym_to_idx(smiles)
	reps = []
	for s in smiles:
		mol = Chem.MolFromSmiles(s)
		atoms = mol.GetAtoms()
		bonds = mol.GetBonds()
		nodes = [a.GetSymbol() for a in atoms] + [b.GetBondType() for b in bonds]
		idxs = [sym_to_idx[n] for n in nodes]
		adj = adjacency_matrix(mol)
		reps.append((idxs, adj))
	return reps, sym_to_idx

def simple_gcn():
	g = MolGraph('CN1CCC[C@H]1c2cccnc2', EMB_SIZE)
	data = g.get_initial() # Random vector for each atom/bond
	adj = g.get_adj_mat()
	num_nodes, vec_size = data.shape
	model = GCN(EMB_SIZE)
	out = model(data,adj)
	print(out)

def gcn_drug_embedder():
	reps, sym_to_idx = preprocess()
	vocab_size = len(sym_to_idx)
	model = GCN_Drug_Embedder(vocab_size, EMB_SIZE)
	for idx, adj in reps[:10]:
		out = model(idx, adj)
		print(out)

def main():
	gcn_drug_embedder()
	
if __name__ == '__main__':
	main()
	