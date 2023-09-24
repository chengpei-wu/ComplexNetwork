# shuffle adjacency matrix
# exchange rows and columns, simultaneously and randomly
import random
from tkinter import Image

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def adj_shuffle(adj: np.ndarray):
    """
    shuffle the adjacency matrix

    """
    s_adj = np.array(adj)
    terms = len(adj) // 2
    while terms:
        index1, index2 = random.randint(
            0, len(adj) - 1), random.randint(0, len(adj) - 1)
        # exchange rows
        s_adj[[index1, index2], :] = s_adj[[index2, index1], :]
        # exchange columns
        s_adj[:, [index1, index2]] = s_adj[:, [index2, index1]]
        terms -= 1
    return s_adj


def adj_sort(adj: np.ndarray):
    """
    sort adjacency matrix
    using node degree rank, top to bottom

    """
    G = nx.from_numpy_matrix(adj)
    degrees = list(nx.degree(G))
    rank_degree = sorted(degrees, key=lambda x: x[1], reverse=True)
    rank_id = [i[0] for i in rank_degree]
    adj = np.array(adj)
    t_adj = adj[rank_id, :]
    t_adj = t_adj[:, rank_id]
    t_adj = csr_matrix(t_adj)
    return t_adj


def bi_linear_sampling(adj: np.ndarray, fixed_size: int):
    """
    using Bilinear interpolation to sampling adjacency matrix,
    to get a fixed size adjacency matrix
    """
    image = Image.new('L', (fixed_size, fixed_size))
    image.paste(Image.fromarray(np.uint8(adj)))
    resized_image = image.resize((fixed_size, fixed_size), Image.BILINEAR)
    result_matrix = np.array(resized_image, dtype=np.int)
    t_adj = csr_matrix(result_matrix)
    # print(result_matrix)
    return t_adj


def random_sampling(adj: np.ndarray, fixed_size: int):
    """
    random sampling for adjacency matrix

    """
    isd = 0
    size = len(adj)
    if isd:
        G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_matrix(adj)
    if size < fixed_size:
        n = fixed_size - size
        while n:
            G.add_node(size + n - 1)
            n -= 1
        A = nx.adjacency_matrix(G).todense()
        s_adj = csr_matrix(adj_shuffle(A))
    else:
        n = size - fixed_size
        rm_ids = np.random.choice(list(G.nodes()), size=(n), replace=False)
        G.remove_nodes_from(rm_ids)
        A = nx.adjacency_matrix(G).todense()
        s_adj = csr_matrix(adj_shuffle(A))
    return s_adj
