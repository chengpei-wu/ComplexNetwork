import random
from typing import Union

import networkx as nx
import numpy as np


def erdos_renyi_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> Union[
    nx.Graph, nx.DiGraph]:
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for i in range(num_nodes):
        G.add_node(i)

    edge_list = []

    while len(edge_list) < num_edges:
        nodes = random.sample(range(num_nodes), 2)
        if not is_directed:
            edge = tuple(sorted(nodes))
        else:
            edge = tuple(nodes)
        if edge not in edge_list:
            edge_list.append(edge)

    for edge in edge_list:
        if is_weighted:
            weight = random.random()
            G.add_edge(*edge, weight=weight)
        else:
            G.add_edge(*edge)
    return G


def barabasi_albert_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    # calculate parameters
    n0 = int(np.ceil(num_edges / num_nodes) + 1)  # initial nodes
    n_add = num_nodes - n0  # rest nodes are added later on
    m_add = num_edges - np.math.comb(n0, 2)  # rest edges are added ...
    m = int(np.floor(m_add / n_add))  # edges to add per-generation
    delta = m_add % n_add
    ms = m * np.ones(num_nodes, dtype=int)
    ms[0:n0] = 0
    if delta:
        pos = np.arange(n0, n0 + delta)
        ms[pos] = ms[pos] + 1

    # create initial graph
    init_graph = nx.complete_graph(n0)
    # add rest nodes and edges by preferential attachment mechanism
    for m in ms[n0:]:
        degrees = dict(init_graph.degree())
        degree_sum = sum(degrees.values())
        new_node = len(init_graph)
        init_graph.add_node(new_node)
        # using degree as weight
        nodes = list(degrees.keys())
        weights = [degree / degree_sum for degree in degrees.values()]
        selected_nodes = np.random.choice(nodes, size=m, p=weights, replace=False)
        for selected_node in selected_nodes:
            init_graph.add_edge(new_node, selected_node)

    # check if directed
    if is_directed:
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(init_graph.nodes())
        for u, v in init_graph.edges():
            if random.random() < 0.5:
                directed_graph.add_edge(u, v)
            else:
                directed_graph.add_edge(v, u)
        init_graph = directed_graph
    # check if weighted
    if is_weighted:
        for u, v in init_graph.edges():
            weight = random.random()
            init_graph[u][v]['weight'] = weight

    return init_graph
