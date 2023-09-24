import random
from copy import deepcopy
from typing import Union

import networkx as nx

from cnt.robustness_optimization.GA.individual import Individual


def make_crossover(ind1: Individual, ind2: Individual, p_cross: float) -> Union[nx.Graph, nx.DiGraph]:
    """

    Parameters
    ----------
    ind1 : individual 1
    ind2 : individual 2
    p_cross : the possibility of cross

    Returns
    -------
    the new crossed graph
    """
    g1 = deepcopy(ind1.g)
    g2 = deepcopy(ind2.g)
    N = g1.number_of_nodes()
    deg_ori = []
    for i in range(N):
        deg_ori.append(g1.degree[i])
    for lab in range(N):
        cross_flag = 1
        if random.random() <= p_cross:
            diff2y_1n = list(set(g2[lab]).difference(set(g1[lab])))  # Found in g2 but not in g1
            diff1y_2n = list(set(g1[lab]).difference(set(g2[lab])))  # Found in g1 but not in g2
            if len(diff2y_1n) == 0 or len(diff1y_2n) == 0:
                continue
            random.shuffle(diff2y_1n)  # shuffle
            random.shuffle(diff1y_2n)
            node_1 = diff2y_1n[0]  # add in g1 but del in g2
            node_2 = diff1y_2n[0]  # add in g2 but del in g1
            flag = 0
            node3in1 = []
            node3in2 = []
            for i in g1[node_1]:
                if node_2 not in g1[i]:
                    node3in1.append(i)
            for i in g2[node_2]:
                if node_1 not in g2[i]:
                    node3in2.append(i)
            if len(node3in1) == 0:
                continue
            random.shuffle(node3in1)
            g1.remove_edge(node_2, lab)
            g1.add_edge(node_1, lab)
            if (node_2 == node3in1[0]):
                continue
            g1.remove_edge(node_1, node3in1[0])
            g1.add_edge(node_2, node3in1[0])
            for i in range(N):
                if deg_ori[i] != g1.degree[i]:
                    cross_flag = 0
            if cross_flag == 0:
                g1 = deepcopy(ind1.g)
            if len(node3in2) == 0:
                continue
            random.shuffle(node3in2)
            g2.remove_edge(node_1, lab)
            g2.add_edge(node_2, lab)
            g2.remove_edge(node_2, node3in2[0])
            if (node_1 == node3in2[0]):
                continue
            g2.add_edge(node_1, node3in2[0])
            for i in range(N):
                if deg_ori[i] != g2.degree[i]:
                    cross_flag = 0
            if cross_flag == 0:
                g2 = deepcopy(ind2.g)
    if random.random() <= 0.5:
        g_get = deepcopy(g1)
    else:
        g_get = deepcopy(g2)
    return g_get
