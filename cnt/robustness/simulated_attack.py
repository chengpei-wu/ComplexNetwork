from copy import deepcopy
from typing import Tuple, List
from typing import Union

import networkx as nx
import numpy as np

from cnt.robustness.network_attack import network_attack


def connectivity_robustness(graph: Union[nx.Graph, nx.DiGraph], attack: str = 'node', strategy: str = 'degree') -> \
        Tuple[List[float], np.ndarray]:
    """
    connectivity robustness of network

    Parameters
    ----------
    graph : the attacked network
    attack : node or edge attacks
    strategy : the strategy of choosing targets under attacks

    Returns
    -------
    connectivity curve under attacks, and its average value

    """
    G = deepcopy(graph)
    N = G.number_of_nodes()
    attack_sequence = network_attack(G, attack=attack, strategy=strategy)

    is_directed = nx.is_directed(G)
    if not is_directed:
        largest_cc = max(nx.connected_components(G), key=len)
    else:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    r_0 = len(largest_cc) / N
    robustness_curve = [r_0]

    for i, target in enumerate(attack_sequence):
        if attack == 'node':
            G.remove_node(target)
            if not is_directed:
                largest_cc = max(nx.connected_components(G), key=len)
            else:
                largest_cc = max(nx.weakly_connected_components(G), key=len)
            r_i = len(largest_cc) / (N - i - 1)

            # another calculation method:
            # r_i = len(largest_cc) / N
        elif attack == 'edge':
            G.remove_edge(*target)
            if not is_directed:
                largest_cc = max(nx.connected_components(G), key=len)
            else:
                largest_cc = max(nx.weakly_connected_components(G), key=len)
            r_i = len(largest_cc) / N
        else:
            raise AttributeError(f'Attack : {attack}, NOT Implemented.')
        robustness_curve.append(r_i)

    return robustness_curve, np.mean(robustness_curve)


def controllability_robustness(graph: Union[nx.Graph, nx.DiGraph], attack: str = 'node', strategy: str = 'degree') -> \
        Tuple[List[float], np.ndarray]:
    """
    controllability robustness of network

    Parameters
    ----------
    graph : the attacked network
    attack : node or edge attacks
    strategy : the strategy of choosing targets under attacks

    Returns
    -------
    controllability curve under attacks, and its average value

    """
    G = deepcopy(graph)
    N = G.number_of_nodes()
    attack_sequence = network_attack(G, attack=attack, strategy=strategy)

    rank_adj = np.linalg.matrix_rank(nx.to_numpy_matrix(G))
    r_0 = max(1, N - rank_adj) / N
    robustness_curve = [r_0]

    for i, target in enumerate(attack_sequence):
        if attack == 'node':
            G.remove_node(target)
            rank_adj = np.linalg.matrix_rank(nx.to_numpy_matrix(G))
            r_i = max(1, (N - i - 1) - rank_adj) / (N - i - 1)
        elif attack == 'edge':
            G.remove_edge(*target)
            rank_adj = np.linalg.matrix_rank(nx.to_numpy_matrix(G))
            r_i = max(1, N - rank_adj) / N
        else:
            raise AttributeError(f'Attack : {attack}, NOT Implemented.')
        robustness_curve.append(r_i)

    return robustness_curve, np.mean(robustness_curve)


def communicability_robustness(graph: Union[nx.Graph, nx.DiGraph], attack: str = 'node', strategy: str = 'degree') -> \
        Tuple[List[float], np.ndarray]:
    """
    communicability robustness of network

    Parameters
    ----------
    graph : the attacked network
    attack : node or edge attacks
    strategy : the strategy of choosing targets under attacks

    Returns
    -------
    communicability curve under attacks, and its average value

    """

    G = deepcopy(graph)
    N = G.number_of_nodes()
    attack_sequence = network_attack(G, attack=attack, strategy=strategy)

    is_directed = nx.is_directed(G)
    if not is_directed:
        connected_components = nx.connected_components(G)
        len_connected_components = np.array([len(i) for i in connected_components])
    else:
        connected_components = nx.weakly_connected_components(G)
        len_connected_components = [len(i) for i in connected_components]
    r_0 = np.sum(len_connected_components * len_connected_components) / (N * N)
    robustness_curve = [r_0]

    for i, target in enumerate(attack_sequence):
        if attack == 'node':
            G.remove_node(target)
            if not is_directed:
                connected_components = nx.connected_components(G)
                len_connected_components = np.array([len(i) for i in connected_components])
            else:
                connected_components = nx.weakly_connected_components(G)
                len_connected_components = [len(i) for i in connected_components]
            r_i = np.sum(len_connected_components * len_connected_components) / (N * N)

            # another calculation method:
            # r_i = len(largest_cc) / N
        elif attack == 'edge':
            G.remove_edge(*target)
            if not is_directed:
                connected_components = nx.connected_components(G)
                len_connected_components = np.array([len(i) for i in connected_components])
            else:
                connected_components = nx.weakly_connected_components(G)
                len_connected_components = [len(i) for i in connected_components]
            r_i = np.sum(len_connected_components * len_connected_components) / (N * N)
        else:
            raise AttributeError(f'Attack : {attack}, NOT Implemented.')
        robustness_curve.append(r_i)

    return robustness_curve, np.mean(robustness_curve)
