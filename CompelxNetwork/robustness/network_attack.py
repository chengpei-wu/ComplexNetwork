import random
from copy import deepcopy
from typing import Union

import networkx as nx


def network_attack(graph: Union[nx.Graph, nx.DiGraph], attack: str = 'node', strategy: str = 'degree') -> list:
    """
    network attack, under a certain network attack strategy.

    Parameters
    ----------
    graph : the graph to be attacked
    attack : node or edge attacks
    strategy : the strategy of choosing targets under attacks

    Returns
    -------
    the attack sequence of nodes and edges
    """
    if attack == 'node':
        return node_attack(graph, strategy=strategy)
    elif attack == 'edge':
        return edge_attack(graph, strategy=strategy)
    else:
        raise AttributeError(f'Attack : {attack}, NOT Implemented.')


def node_attack(graph: Union[nx.Graph, nx.DiGraph], strategy: str = 'degree') -> list:
    """
        node attacks, under a certain strategy.

        Parameters
        ----------
        graph : the graph to be attacked
        strategy : the strategy of choosing targets under node removals

        Returns
        -------
        the attack sequence of nodes
    """
    sequence = []
    G = deepcopy(graph)
    N = G.number_of_nodes()
    for _ in range(N - 1):
        if strategy == 'degree':
            degrees = dict(nx.degree(G))
            _node = max(degrees, key=degrees.get)
        elif strategy == 'random':
            _node = random.sample(list(G.nodes), 1)[0]
        elif strategy == 'betweenness':
            bets = dict(nx.betweenness_centrality(G))
            _node = max(bets, key=bets.get)
        else:
            raise AttributeError(f'Attack strategy: {strategy}, NOT Implemented.')
        G.remove_node(_node)
        sequence.append(_node)
    return sequence


def edge_attack(graph: Union[nx.Graph, nx.DiGraph], strategy: str = 'random') -> list:
    """
            edge attacks, under a certain strategy.

            Parameters
            ----------
            graph : the graph to be attacked
            strategy : the strategy of choosing targets under edge removals

            Returns
            -------
            the attack sequence of edges
    """
    sequence = []
    G = deepcopy(graph)
    M = G.number_of_edges()
    for _ in range(M):
        if strategy == 'random':
            _edge = random.sample(list(G.edges), 1)[0]
        else:
            raise AttributeError(f'Attack strategy: {strategy}, NOT Implemented.')
        G.remove_edge(*_edge)
        sequence.append(_edge)
    return sequence
