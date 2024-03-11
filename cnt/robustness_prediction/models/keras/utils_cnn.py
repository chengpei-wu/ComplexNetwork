import random

import networkx as nx
import numpy as np
import scipy.io as sio
import tensorflow.keras.callbacks
from scipy.sparse import csr_matrix
from tqdm import tqdm


def load_network(path, label):
    mat = sio.loadmat(path)
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    A = []
    y = []
    for i in range(len_net):
        for j in tqdm(range(len_instance), desc=f'loading networks (type {i}): '):
            s_adj = csr_matrix(adj_shuffle(networks[i, j]['adj'][0][0].todense()))
            A.append(s_adj)
            y.append(networks[i, j][label][0][0])
    y = np.array(y)
    A = np.array(A)
    return A, y


def load_network_with_fixed_size(path, label, fixed_size):
    mat = sio.loadmat(path)
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    A = []
    y = []
    for i in range(len_net):
        for j in tqdm(range(len_instance), desc=f'loading networks (type {i}): '):
            adj = networks[i, j]['adj'][0][0]
            if adj.shape[0] != fixed_size:
                adj = fix_adj_size(adj.todense(), fixed_size)
            A.append(adj)
            y.append(networks[i, j][label][0][0])
    y = np.array(y)
    A = np.array(A)
    return A, y


def load_labels(path, label='lc'):
    mat = sio.loadmat(path)
    dataY = []
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    for i in range(len_net):
        for j in tqdm(range(len_instance), desc=f'loading network labels (type {i}): '):
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


def fix_adj_size(adj, fixed_size):
    isd = 0
    size = len(adj)
    if isd:
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(adj)
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


class Save_loss(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append([logs.get('mae'), logs.get('val_mae')])
        if len(self.losses) > 1:
            print(f'improved: {self.losses[-1][0] - self.losses[-2][0]}')
        print(self.losses)
