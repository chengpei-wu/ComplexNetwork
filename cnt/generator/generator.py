import random
from typing import Union

import networkx as nx
import numpy as np


def erdos_renyi_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> Union[
    nx.Graph, nx.DiGraph]:
    """
    return Erdos Renyi (ER) graph.

    Parameters
    ----------
    num_nodes : number of nodes
    num_edges : number of edges
    is_directed : return directed graph
    is_weighted : return graph with random edge weight

    Returns
    -------
    nx.Graph or nx.DiGraph

    References
    ----------
    .. [1] P. Erdos and A. Renyi, "On the strength of connectedness of a random
        graph," Acta Mathematica Hungarica, vol. 12, no. 1-2, pp. 261–267, 1964.
    """
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
    """
        return Barabasi Albert (BA) graph.

        Parameters
        ----------
        num_nodes : number of nodes
        num_edges : number of edges
        is_directed : return directed graph
        is_weighted : return graph with random edge weight

        Returns
        -------
        nx.Graph or nx.DiGraph

        References
        ----------
        .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
            random networks", Science 286, pp 509-512, 1999.
        """
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


def generic_scale_free_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    raise NotImplementedError('the model is not implemented yet.')


def extremely_homogeneous_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    raise NotImplementedError('the model is not implemented yet.')


def multi_local_world_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    raise NotImplementedError('the model is not implemented yet.')


def q_snapback_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    raise NotImplementedError('the model is not implemented yet.')


def random_hexagon_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    raise NotImplementedError('the model is not implemented yet.')


def random_triangle_graph(num_nodes: int, num_edges: int, is_directed: bool = False, is_weighted: bool = False) -> \
        Union[
            nx.Graph, nx.DiGraph]:
    raise NotImplementedError('the model is not implemented yet.')


# def newman_watts_samll_world_graph(num_nodes: int, num_edges: int, p: float, is_directed: bool = False,
#                                    is_weighted: bool = False) -> \
#         Union[
#             nx.Graph, nx.DiGraph]:
#     graph = nx.newman_watts_strogatz_graph(num_nodes, 2 * num_edges // num_nodes, p)
#
#     if is_directed:
#         directed_graph = nx.DiGraph()
#         # 将无向图的边赋予随机方向
#         for edge in graph.edges():
#             if random.random() < 0.5:
#                 directed_graph.add_edge(edge[0], edge[1])
#             else:
#                 directed_graph.add_edge(edge[1], edge[0])
#         graph = directed_graph
#     # check if weighted
#     if is_weighted:
#         for u, v in graph.edges():
#             weight = random.random()
#             graph[u][v]['weight'] = weight
#
#     return graph
#
#
# def watts_strogatz_samll_world_graph(num_nodes: int, num_edges: int, p: float, is_directed: bool = False,
#                                      is_weighted: bool = False) -> \
#         Union[
#             nx.Graph, nx.DiGraph]:
#     graph = nx.watts_strogatz_graph(num_nodes, 2 * num_edges // num_nodes, p)
#
#     if is_directed:
#         directed_graph = nx.DiGraph()
#         # 将无向图的边赋予随机方向
#         for edge in graph.edges():
#             if random.random() < 0.5:
#                 directed_graph.add_edge(edge[0], edge[1])
#             else:
#                 directed_graph.add_edge(edge[1], edge[0])
#         graph = directed_graph
#     # check if weighted
#     if is_weighted:
#         for u, v in graph.edges():
#             weight = random.random()
#             graph[u][v]['weight'] = weight
#
#     return graph


def network_with_degree_distribution(degree_distribution: str, num_nodes: int, num_edges: int):
    """
    Parameters
    ----------
    degree_distribution : the network degree distribution
    num_nodes : number of nodes
    num_edges : number of edges

    Returns
    -------
    an undirected network with specified degree distribution

    """

    G = nx.Graph()

    nodes = range(num_nodes)
    G.add_nodes_from(nodes)

    if degree_distribution == "power-law":
        sfpara = {
            'theta': 0,
            'mu': 0.999
        }
        random.seed()
        w = [(i + sfpara['theta']) ** -sfpara['mu'] for i in range(1, num_nodes + 1)]
        ransec = np.cumsum(w)
    elif degree_distribution == "poisson":
        lam = 5
        w = np.exp(-lam) * lam ** np.arange(num_nodes) / np.math.factorial(np.arange(num_nodes))
        ransec = np.cumsum(w)
    else:
        raise NotImplementedError(f'{degree_distribution}')

    while G.number_of_edges() < num_edges:
        r = random.random() * ransec[-1]
        node1 = next((i for i, v in enumerate(ransec) if r <= v), None)
        r = random.random() * ransec[-1]
        node2 = next((i for i, v in enumerate(ransec) if r <= v), None)

        if node1 is not None and node2 is not None and node1 != node2:
            G.add_edge(node1, node2)

    return G
