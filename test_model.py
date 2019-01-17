import math
import csv
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F

from rdkit import Chem

from graph_rep import MolGraph
from gcn import GCN


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

def basic_fingerprint_embedding():
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

def simple_gcn():
	g = MolGraph('CN1CCC[C@H]1c2cccnc2')
	g.create_fingerprint()
	data = g.get_initial()
	num_nodes, vec_size = data.shape
	model = GCN(vec_size, 5, 5, g)
	out = model(data)
	print(out)

def main():
	simple_gcn()
	
if __name__ == '__main__':
	main()
	