import random

import networkx as nx


def missing_nodes(adj, strategy, rate):
    if rate == 0:
        return adj
    isd = adj != adj.T
    if strategy[1:] == 'random':
        missing_adj = remove_random_nodes(adj, rate, isd)
    else:
        missing_adj = None
    return missing_adj


def missing_edges(adj, strategy, rate):
    if rate == 0:
        return adj
    isd = adj != adj.T
    if strategy[1:] == 'rnd':
        missing_adj = remove_random_edges(adj, rate, isd)
    else:
        missing_adj = None
    return missing_adj


def remove_random_edges(adj, rate, isd):
    if isd:
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(adj)
    number_rm_edges = round(rate * G.number_of_edges())
    for i in range(number_rm_edges):
        all_edges = G.edges()
        rm_id = random.randint(0, G.number_of_edges() - 1)
        rm_edge = all_edges[rm_id]
        G.remove_edge(rm_edge[0], rm_edge[1])
    missing_adj = nx.adjacency_matrix(G)
    return missing_adj


def remove_random_nodes(adj, rate, isd):
    if isd:
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(adj)
    number_rm_nodes = round(rate * len(adj))
    for i in range(number_rm_nodes):
        rm_id = random.randint(0, G.number_of_nodes() - 1)
        G.remove_node(rm_id)
    missing_adj = nx.adjacency_matrix(G)
    return missing_adj
