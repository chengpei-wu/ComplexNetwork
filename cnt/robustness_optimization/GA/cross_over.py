import random
from copy import deepcopy

import networkx as nx

from .individual import Individual


def make_crossover(ind1: Individual, ind2: Individual, p_cross: float):
    G1 = deepcopy(ind1.g)
    G2 = deepcopy(ind2.g)
    for node_i in G1.nodes:
        if random.random() <= p_cross:
            unique_neighbors_g1 = list(set(nx.neighbors(G1, node_i)).difference(set(nx.neighbors(G2, node_i))))
            unique_neighbors_g2 = list(set(nx.neighbors(G2, node_i)).difference(set(nx.neighbors(G1, node_i))))

            if len(unique_neighbors_g1) == 0 or len(unique_neighbors_g2) == 0:
                continue

            for node_j in unique_neighbors_g1:
                if len(unique_neighbors_g2) == 0:
                    break

                node_k = random.choice(unique_neighbors_g2)

                node_ls = nx.neighbors(G1, node_k)

                for l in node_ls:
                    if l != node_k and l != node_i and l != node_j and G1.has_edge(l, node_k) and not G1.has_edge(l, node_j) \
                            and G1.has_edge(node_i, node_j) and not G1.has_edge(node_i, node_k):
                        node_l = l
                        G1.remove_edge(node_l, node_k)
                        G1.add_edge(node_l, node_j)
                        G1.remove_edge(node_i, node_j)
                        G1.add_edge(node_i, node_k)
                        break

                node_ms = nx.neighbors(G2, node_j)
                for m in node_ms:
                    if m != node_j and m != node_i and m != node_k and G2.has_edge(m, node_j) and not G2.has_edge(m, node_k) \
                            and G2.has_edge(node_i, node_k) and not G2.has_edge(node_i, node_j):
                        node_m = m
                        G2.remove_edge(node_m, node_j)
                        G2.add_edge(node_m, node_k)
                        G2.remove_edge(node_i, node_k)
                        G2.add_edge(node_i, node_j)
                        break
                unique_neighbors_g2.remove(node_k)
    if random.random() <= 0.5:
        g_get = deepcopy(G1)
    else:
        g_get = deepcopy(G2)
    return g_get
