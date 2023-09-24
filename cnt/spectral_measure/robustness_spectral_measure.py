from typing import Union

import networkx as nx
import numpy as np


def spectral_radius(graph: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    """
    spectral radius is the largest eigenvalue of graph adjacency matrix
    """
    return np.max(np.linalg.eigvals(nx.to_numpy_matrix(graph))).real


def spectral_gap(graph: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    """
    spectral gap is the difference of top-2 large eigenvalues of graph adjacency matrix
    """
    eig_val = np.linalg.eigvals(nx.to_numpy_matrix(graph))
    max_eig_val = np.sort(eig_val)[-1::-1]
    return (max_eig_val[0] - max_eig_val[1]).real


def natural_connectivity(graph: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    """
        natural connectivity
    """
    eig_val = np.linalg.eigvals(nx.to_numpy_matrix(graph))
    return np.log(np.sum(np.exp(eig_val) / graph.number_of_nodes())).real


def algebraic_connectivity(graph: Union[nx.Graph, nx.DiGraph]) -> float:
    """
        algebraic connectivity is the second smallest eigenvalue of graph laplacian matrix
    """
    return nx.algebraic_connectivity(graph)


def effective_resistance(graph: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    """
    effective resistance

    """
    eig_val = np.sort(np.linalg.eigvals(nx.laplacian_matrix(graph).todense()))
    return (graph.number_of_nodes() * np.sum(1 / eig_val[1:])).real


def spanning_tree_count(graph: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    """
    spanning tree count

    """

    eig_val = np.sort(np.linalg.eigvals(nx.laplacian_matrix(graph).todense()))
    return (np.sum(eig_val[1:]) / graph.number_of_nodes()).real


def assortativity(graph: Union[nx.Graph, nx.DiGraph]) -> float:
    """
    assortativity measures the similarity of connections
    in the graph with respect to the node degree.
    """
    return nx.degree_assortativity_coefficient(graph)
