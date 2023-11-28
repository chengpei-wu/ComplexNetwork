from copy import deepcopy

import networkx as nx


def calculate_robustness_wang(graph):
    R = 0
    N = graph.number_of_nodes()
    G = deepcopy(graph)
    for atk in range(N):
        deg = [G.degree(i) for i in range(N)]
        atk_ind = deg.index(max(deg))
        '''Get current node with largest degree'''
        dele = []
        for i in G[atk_ind]:
            dele.append(i)
        for i in dele:
            G.remove_edge(i, atk_ind)
        largest_cc = max(nx.connected_components(G), key=len)  # The largest connected component
        R = R + float(len(largest_cc)) / N
    return R / N


def calculate_robustness(graph):
    R = 0
    N = graph.number_of_nodes()
    G = deepcopy(graph)
    for _ in range(N):
        degrees = dict(nx.degree(G))
        _node = max(degrees, key=degrees.get)
        G.remove_node(_node)
        if not nx.is_empty(G):
            largest_cc = max(nx.connected_components(G), key=len)
        else:
            largest_cc = []
        R = R + float(len(largest_cc)) / N
    return R / N
