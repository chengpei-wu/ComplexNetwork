from typing import Union

import networkx as nx
import numpy as np


def spectral_radius(graph: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    return np.max(np.linalg.eig(nx.to_numpy_matrix(graph))[0])


def spectral_gap(graph: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    eig_val = np.linalg.eig(nx.to_numpy_matrix(graph))[0]
    max_eig_val = np.sort(eig_val)[-1::]
    return max_eig_val[0] - max_eig_val[1]


def natural_connectivity(graph: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    eig_val = np.linalg.eig(nx.to_numpy_matrix(graph))[0]
    return np.log(np.sum(np.exp(eig_val) / graph.number_of_nodes()))


def algebraic_connectivity(graph: Union[nx.Graph, nx.DiGraph]) -> float:
    return nx.algebraic_connectivity(graph)


def effective_resistance(graph: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    eig_val = np.linalg.eig(nx.laplacian_matrix(graph))[0]
    return graph.number_of_nodes() * np.sum(1 / eig_val[1:])


def spanning_tree_count(graph: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    eig_val = np.linalg.eig(nx.laplacian_matrix(graph))[0]
    return np.sum(eig_val[1:]) / graph.number_of_nodes()


def assortativity(graph: Union[nx.Graph, nx.DiGraph]) -> float:
    return nx.degree_assortativity_coefficient(graph)
