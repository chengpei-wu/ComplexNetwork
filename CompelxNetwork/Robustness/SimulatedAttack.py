from copy import deepcopy

import networkx as nx
import numpy as np

from Attack import Attack


def ConnectivityRobustness(graph, attack='node', strategy='degree'):
    G = deepcopy(graph)
    N = G.number_of_nodes()
    attack_sequence = Attack(G, attack=attack, strategy=strategy)

    largest_cc = max(nx.connected_components(G), key=len)
    r_0 = len(largest_cc) / N
    robustness_curve = [r_0]

    for i, target in enumerate(attack_sequence):
        if attack == 'node':
            G.remove_node(target)
            largest_cc = max(nx.connected_components(G), key=len)
            r_i = len(largest_cc) / (N - i - 1)

            # another calculation method:
            # r_i = len(largest_cc) / N
        elif attack == 'edge':
            G.remove_edge(*target)
            largest_cc = max(nx.connected_components(G), key=len)
            r_i = len(largest_cc) / N
        else:
            raise AttributeError(f'Attack : {attack}, NOT Implemented.')
        robustness_curve.append(r_i)

    return robustness_curve, np.mean(robustness_curve)


# todo: ControllabilityRobustness

def ControllabilityRobustness(graph, attack='node', strategy='degree'):
    pass


# todo: CommunicabilityRobustness
def CommunicabilityRobustness(graph, attack='node', strategy='degree'):
    pass
