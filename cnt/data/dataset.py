import pickle
import random
from typing import Union, List

import dgl
import numpy as np
from dgl.data import DGLDataset
from tqdm import tqdm

import cnt
from cnt.generator import erdos_renyi_graph, barabasi_albert_graph
from cnt.robustness import connectivity_robustness, controllability_robustness, \
	communicability_robustness


def create_network_instances(topology_type: str, is_directed: bool, is_weighted: bool, num_instance: int,
							 network_size: Union[int, tuple],
							 average_degree: Union[float, int, tuple]) -> list:
	"""
	generating synthetic network instances

	Parameters
	----------
	topology_type : the type of synthetic network
	is_directed : directed network or not
	is_weighted : random weighted network or not
	num_instance : number of instances of the specified synthetic network type
	network_size : network size (node numbers)
	average_degree : average degree of synthetic networks

	Returns
	-------
	a list of generated synthetic network instances

	"""
	networks = []
	for _ in tqdm(range(num_instance), desc=f'generating {topology_type.upper()} networks:'):
		if isinstance(network_size, tuple):
			num_nodes = random.randint(*network_size)
		else:
			num_nodes = network_size

		if isinstance(average_degree, tuple):
			k = random.uniform(*average_degree)
			if is_directed:
				num_edges = round(k * num_nodes)
			else:
				num_edges = round((k * num_nodes) / 2)
		else:
			k = average_degree
			if is_directed:
				num_edges = round(k * num_nodes)
			else:
				num_edges = round((k * num_nodes) / 2)

		if topology_type == 'er':
			graph = erdos_renyi_graph(
				num_nodes=num_nodes,
				num_edges=num_edges,
				is_directed=is_directed,
				is_weighted=is_weighted
			)
		elif topology_type == 'ba':
			graph = barabasi_albert_graph(
				num_nodes=num_nodes,
				num_edges=num_edges,
				is_directed=is_directed,
				is_weighted=is_weighted
			)
		else:
			raise NotImplementedError
		networks.append(graph)
	return networks


def save_simulated_network_dataset(
		topology_types: List[str], is_directed: bool, is_weighted: bool,
		num_instance: int,
		network_size: Union[int, tuple],
		average_degree: Union[float, int, tuple],
		**kwargs
):
	"""
	generating and saving synthetic network dataset

	Parameters
	----------
	topology_types : a list of the types of synthetic networks
	is_directed : directed network or not
	is_weighted : random weighted network or not
	num_instance : number of instances of each synthetic network type
	network_size : network size (node numbers)
	average_degree : average degree
	kwargs :

	Returns
	-------
	a dict of generated networks with simulation results (e.g., network robustness)

	"""
	graghs = []
	graph_labels = []
	connectivity_curves = []
	controllability_curves = []
	communicability_curves = []

	for topology_type in topology_types:
		network_instances = create_network_instances(
			topology_type=topology_type,
			is_directed=is_directed,
			is_weighted=is_weighted,
			num_instance=num_instance,
			network_size=network_size,
			average_degree=average_degree
		)
		for network_instance in tqdm(network_instances,
									 desc=f'calculating robustness of {topology_type.upper()} networks: '):
			graghs.append(network_instance)
			graph_labels.append(topology_type)
			save_robustness = kwargs.get('save_robustness', None)
			if save_robustness:
				for r in save_robustness:
					if r == 'connectivity':
						curve, _ = connectivity_robustness(
							graph=network_instance,
							attack=kwargs.get('attack', 'unknown'),
							strategy=kwargs.get('strategy', 'unknown'),
						)
						connectivity_curves.append(curve)
					elif r == 'controllability':
						curve, _ = controllability_robustness(
							graph=network_instance,
							attack=kwargs.get('attack', 'unknown'),
							strategy=kwargs.get('strategy', 'unknown'),
						)
						controllability_curves.append(curve)
					elif r == 'communicability':
						curve, _ = communicability_robustness(
							graph=network_instance,
							attack=kwargs.get('attack', 'unknown'),
							strategy=kwargs.get('strategy', 'unknown'),
						)
						communicability_curves.append(curve)
					else:
						raise NotImplementedError(f'{r} not implemented')
	res = {
		'graphs': graghs,
		'labels': graph_labels,
	}

	if 'connectivity' in kwargs.get('save_robustness', None):
		res['connectivity_curves'] = connectivity_curves
	if 'controllability' in kwargs.get('save_robustness', None):
		res['controllability_curves'] = controllability_curves
	if 'communicability' in kwargs.get('save_robustness', None):
		res['communicability_curves'] = communicability_curves

	save_path = kwargs.get('save_path', None)
	if save_path:
		with open(f'{save_path}.pkl', 'wb') as file:
			pickle.dump(res, file)
			print(f'Simulated network dataset save at {save_path}.pkl')
	return res


def load_simulated_network_dataset(load_path: str) -> dict:
	"""
	loading simulation network dataset

	Parameters
	----------
	load_path : the dataset path

	Returns
	-------
	a dict of simulation dataset

	"""
	with open(f'{load_path}.pkl', 'rb') as file:
		res = pickle.load(file)
		return res


class DGLSyntheticDataset(DGLDataset):
	"""
		DGLDataset for GNNs on graph tasks.
	"""

	# todo: download dataset to local
	def __init__(self, network_type, average_degree: tuple, num_nodes, num_graphs=1000):
		'''

		Parameters
		----------
		network_type: synthetic network type
		average_degree: average degree
		num_nodes: number of nodes
		num_graphs: number of graphs
		'''
		self.graphs = None
		self.network_type = network_type
		self.average_degree = average_degree
		self.num_nodes = num_nodes
		self.num_graphs = num_graphs
		super().__init__(name=f"SyntheticDataset_{network_type}")

	def process(self):
		graphs = []
		for _ in tqdm(
				range(self.num_graphs), desc=f"generating {self.network_type} nets"
		):
			num_edges = int(
				np.random.uniform(*self.average_degree) * self.num_nodes * 0.5
			)
			if self.network_type == "er":
				nx_g = cnt.erdos_renyi_graph(num_nodes=self.num_nodes, num_edges=num_edges)
			elif self.network_type == "ba":
				# todo
				pass
			else:
				raise NotImplementedError(
					f"network type {self.network_type} not implemented yet."
				)
			_g = dgl.add_self_loop(dgl.from_networkx(nx_g))
			_g.ndata["deg_feat"] = _g.in_degrees().view(-1, 1).float()
			graphs.append(_g)

		self.graphs = graphs

	def __getitem__(self, idx) -> dgl.DGLGraph:
		return self.graphs[idx]

	def __len__(self):
		return len(self.graphs)
