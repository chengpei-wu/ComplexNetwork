import networkx as nx
import numpy as np


def SpectralRadius(graph):
    return np.max(np.linalg.eig(nx.to_numpy_matrix(graph))[0])


def SpectralGap(graph):
    eig_val = np.linalg.eig(nx.to_numpy_matrix(graph))[0]
    max_eig_val = np.sort(eig_val)[-1::]
    return max_eig_val[0] - max_eig_val[1]


def NaturalConnectivity(graph):
    eig_val = np.linalg.eig(nx.to_numpy_matrix(graph))[0]
    return np.log(np.sum(np.exp(eig_val) / graph.number_of_nodes()))


def AlgebraicConnectivity(graph):
    return nx.algebraic_connectivity(graph)


def EffectiveResistance(graph):
    eig_val = np.linalg.eig(nx.laplacian_matrix(graph))[0]
    return graph.number_of_nodes() * np.sum(1 / eig_val[1:])


def SpanningTreeCount(graph):
    eig_val = np.linalg.eig(nx.laplacian_matrix(graph))[0]
    return np.sum(eig_val[1:]) / graph.number_of_nodes()


def Assortativity(graph):
    return nx.degree_assortativity_coefficient(graph)
