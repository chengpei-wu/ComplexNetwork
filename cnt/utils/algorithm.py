import random
from copy import deepcopy

import networkx as nx
import numpy as np


def havel_hakimi_process(degrees: list, p=False) -> list:
    """
    make a sequence to be a valid degree sequence for constructing a simple graph, based on Havel Hakimi algorithm,

    Parameters
    ----------
    degrees : the original degree sequence
    p: print degrees or not

    Returns
    -------
    the valid degree sequence

    """
    degrees.sort(reverse=True)
    degrees = np.array(degrees)
    ori_degrees = deepcopy(degrees)
    cnt = 0
    while len(degrees) > 0:
        if p:
            print('de:', degrees)
        d = degrees[0]
        degrees = degrees[1:]

        if d > len(degrees) > 0:
            ori_degrees[cnt] = len(degrees)

        if d < 0 < len(degrees):
            ori_degrees[cnt] = ori_degrees[cnt] + abs(d)

        if len(degrees) == 0 and d != 0:
            if p:
                print('**', ori_degrees, d)
            ori_degrees[cnt] -= d
            print(ori_degrees)
            break

        minus = 0
        for i in range(min(d, len(degrees))):
            minus += 1
            degrees[i] -= 1
        if minus < d and len(degrees) != 0:
            ori_degrees[cnt] = ori_degrees[cnt] - (d - minus)
        cnt += 1
    return list(ori_degrees)


def uncon2con(graph, method='chain'):
    G = graph.copy()
    original_edges = list(G.edges())
    is_directed = nx.is_directed(G)

    # connecting all connected components
    if is_directed:
        connected_components = list(nx.weakly_connected_components(G))
    else:
        connected_components = list(nx.connected_components(G))
    add_counts = 0
    if len(connected_components) > 1:
        # connecting all connected components using a chain
        if method == 'chain':
            for i in range(len(connected_components) - 1):
                node1 = next(iter(connected_components[i]))
                node2 = next(iter(connected_components[i + 1]))
                G.add_edge(node1, node2)
                add_counts += 1
        # attaching other connected components on the largest connected component
        elif method == 'attach':
            for i in range(len(connected_components) - 1):
                if is_directed:
                    largest_cc = list(max(nx.weakly_connected_components(G), key=len))
                else:
                    largest_cc = list(max(nx.connected_components(G), key=len))
                node1 = random.choice(largest_cc)
                node2 = next(iter(connected_components[i + 1]))
                G.add_edge(node1, node2)
                add_counts += 1

    # delete edges while keeping connectivity
    edges_to_remove = random.sample(original_edges, add_counts * 4)

    for edge in edges_to_remove:
        if add_counts <= 0:
            break
        G_temp = G.copy()
        G_temp.remove_edge(*edge)

        # check connectivity
        if is_directed:
            if nx.is_weakly_connected(G_temp):
                G = G_temp
                add_counts -= 1
        else:
            if nx.is_connected(G_temp):
                G = G_temp
                add_counts -= 1
    return G
