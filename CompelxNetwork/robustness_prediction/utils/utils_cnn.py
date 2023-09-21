import random

import networkx as nx
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from scipy.sparse import csr_matrix

from utils.utils_tool_function import print_progress


# load networks and labels from .mat files
# do NOT change network size (random sampling or others)
def load_network_cnn(path, label):
    mat = sio.loadmat(path)
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    graphs = []
    for i in range(len_net):
        for j in range(len_instance):
            print_progress(i * len_instance + j + 1, len_net * len_instance, prefix='loading_networks:')
            s_adj = csr_matrix(adj_shuffle(networks[i, j]['adj'][0][0].todense()))
            graphs.append({'g': s_adj, 'label': networks[i, j][label][0][0]})
    return graphs


# load networks and labels from .mat files
# fixing the network size (random sampling or others)
def load_var_network(path, label, sampling, fixed_size):
    mat = sio.loadmat(path)
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    graphs = []
    for i in range(len_net):
        for j in range(len_instance):
            print_progress(i * len_instance + j + 1, len_net * len_instance, prefix='loading_networks:')
            adj = networks[i, j]['adj'][0][0]
            if adj.shape[0] != fixed_size:
                if sampling == 'random':
                    adj = random_sampling(adj.todense(), fixed_size)
                elif sampling == 'bi':
                    adj = bi_linear_sampling(adj.todense(), fixed_size)
                else:
                    print(f'The sampling method: {sampling}, has not been implemented.')
            graphs.append({'g': adj, 'label': networks[i, j][label][0][0]})
    return graphs


# only load labels from .mat files
def load_labels(path, label='lc'):
    mat = sio.loadmat(path)
    dataY = []
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    for i in range(len_net):
        for j in range(len_instance):
            print_progress(i * len_instance + j + 1, len_net * len_instance, prefix='loading_networks:')
            if label not in ['pt', 'cc']:
                dataY.append(networks[i, j][label][0][0])
            elif label == 'pt':
                pt = networks[i, j]['pt'][0][0].squeeze()
                adj = networks[i, j]['adj'][0][0]
                peak = (np.mean(np.where(pt == np.amax(pt))) / adj.shape[0])
                dataY.append(peak)
            else:
                pt = networks[i, j]['pt'][0][0].squeeze()
                adj = networks[i, j]['adj'][0][0]
                cc_peak = np.amax(pt) / adj.shape[0]
                dataY.append(cc_peak)
    return np.array(dataY)


# load graph embeddings, produced by Patchy-SAN
def load_lfr_embeddings(path, w=1000):
    X_tensors = []
    try:
        X_tensors = np.load(path)
        samples = len(X_tensors)
        if w == 500:
            X_tensors = X_tensors.reshape(samples, 100, 100, 1)
        if w == 1000:
            X_tensors = X_tensors.reshape(samples, 125, 160, 1)
    except FileNotFoundError:
        print('No lfr_embeddings found!!!\n please generate embeddings first.')
    return X_tensors


# shuffle adjacency matrix
# exchange rows and columns, simultaneously and randomly
def adj_shuffle(adj):
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


# sort adjacency matrix
# using node degree rank, top to bottom
def adj_sort(adj):
    G = nx.from_numpy_matrix(adj)
    degrees = list(nx.degree(G))
    rank_degree = sorted(degrees, key=lambda x: x[1], reverse=True)
    rank_id = [i[0] for i in rank_degree]
    adj = np.array(adj)
    t_adj = adj[rank_id, :]
    t_adj = t_adj[:, rank_id]
    t_adj = csr_matrix(t_adj)
    return t_adj


# using Bilinear interpolation to sampling adjacency matrix,
# to get a fixed size adjacency matrix
def bi_linear_sampling(adj, size):
    image = Image.new('L', (size, size))
    image.paste(Image.fromarray(np.uint8(adj)))
    resized_image = image.resize((size, size), Image.BILINEAR)
    result_matrix = np.array(resized_image, dtype=np.int)
    t_adj = csr_matrix(result_matrix)
    # print(result_matrix)
    return t_adj


# using random node addition and deletion to sampling adjacency matrix,
# to get a fixed size adjacency matrix
def random_sampling(adj, fixed_size):
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


def collate_cnn(samples):
    batch_size = len(samples)
    batch_graphs = [sample['g'].todense() for sample in samples]
    batch_labels = [sample['label'] for sample in samples]
    return torch.tensor(np.array(batch_graphs, dtype=np.float32)).view(
        batch_size, 1, len(batch_graphs[0]), len(batch_graphs[0]), ), torch.tensor(
        np.array(batch_labels, dtype=np.float32)).squeeze().float().view(
        batch_size, -1)
