import random

import networkx as nx


def missing_nodes(adj, strategy, rate):
    """
    add node noise

    Parameters
    ----------
    adj : the adjacency matrix
    strategy :
    rate :

    Returns
    -------

    """
    if rate == 0:
        return adj
    if strategy[1:] == 'rnd':
        missing_adj = remove_random_nodes(adj, rate)
    elif strategy == 'rnd_nbr':
        missing_adj = remove_random_neighbors(adj, rate)
    else:
        missing_adj = None
    return missing_adj


def missing_edges(adj, strategy, rate):
    if rate == 0:
        return adj
    if strategy[1:] == 'rnd':
        missing_adj = remove_random_edges(adj, rate)
    else:
        missing_adj = None
    return missing_adj


def remove_random_edges(adj, rate, isd):
    if isd:
        G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_matrix(adj)
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
        G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_matrix(adj)
    number_rm_nodes = round(rate * len(adj))
    for i in range(number_rm_nodes):
        rm_id = random.randint(0, G.number_of_nodes() - 1)
        G.remove_node(rm_id)
    missing_adj = nx.adjacency_matrix(G)
    return missing_adj


def remove_random_neighbors(adj, rate, isd):
    # remove nodes base on BFS start with a random node
    if isd:
        G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_matrix(adj)
    number_rm_nodes = round(rate * len(adj))
    rm_id = random.randint(0, G.number_of_nodes() - 1)
    # print(f'find random start node: {rm_id}')
    rm_ids = set()
    front, rear = -1, -1
    rm_queue = [-1] * G.number_of_nodes() ** 2
    front += 1
    rm_queue[front] = rm_id
    while rear < front and number_rm_nodes > 0:
        rear += 1
        n = rm_queue[rear]
        if not rm_ids.__contains__(n):
            rm_ids.add(n)
            # print(f'will remove node: {n}')
            number_rm_nodes -= 1
        n_nbrs = list(G.neighbors(n))
        # print(f'find neighbors of node {n} : {n_nbrs}')
        if not n_nbrs:
            # print(f'node {n} has no neighbors')
            while True:
                t = random.randint(0, G.number_of_nodes() - 1)
                if t != n:
                    front += 1
                    rm_queue[front] = t
                    # print(f'find a new random node {t} as start node')
                    break
        else:
            for i in n_nbrs:
                front += 1
                rm_queue[front] = i
    # remove nodes
    for i in rm_ids:
        G.remove_node(i)
    missing_adj = nx.adjacency_matrix(G)
    # return new adj
    return missing_adj
