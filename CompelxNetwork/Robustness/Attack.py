import random
from copy import deepcopy

import networkx as nx


def Attack(graph, attack='node', strategy='degree'):
    if attack == 'node':
        return NodeAttack(graph, strategy=strategy)
    elif attack == 'edge':
        return EdgeAttack(graph, strategy=strategy)
    else:
        raise AttributeError(f'Attack : {attack}, NOT Implemented.')


def NodeAttack(graph, strategy='degree'):
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


def EdgeAttack(graph, strategy='random'):
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
