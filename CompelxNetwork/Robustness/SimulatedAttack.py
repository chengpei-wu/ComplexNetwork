from copy import deepcopy

import networkx as nx
import numpy as np

from Attack import Attack


def ConnectivityRobustness(graph, attack='node', strategy='degree'):
    G = deepcopy(graph)
    N = G.number_of_nodes()
    attack_sequence = Attack(G, attack=attack, strategy=strategy)

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


def ControllabilityRobustness(graph, attack='node', strategy='degree'):
    G = deepcopy(graph)
    N = G.number_of_nodes()
    attack_sequence = Attack(G, attack=attack, strategy=strategy)

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


def CommunicabilityRobustness(graph, attack='node', strategy='degree'):
    G = deepcopy(graph)
    N = G.number_of_nodes()
    attack_sequence = Attack(G, attack=attack, strategy=strategy)

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
