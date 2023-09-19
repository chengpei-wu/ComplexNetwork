import networkx as nx
import numpy as np
import scipy.io as sio

from models.CNN_LFR.embedding import LFR
from utils.utils_gnn import print_progress


def get_lfr_tensors(data_path, save_path):
    mat = sio.loadmat(data_path)
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    tensors = []
    temp_adj = networks[0, 0]['adj'][0, 0].todense()
    if np.array_equal(temp_adj.T,temp_adj):
        isd = 0
    else:
        isd = 1
    for i in range(len_net):
        for j in range(len_instance):
            print_progress(i * len_instance + j + 1, len_net * len_instance, prefix='saving lfr embeddings:', )
            adj = networks[i, j]['adj'][0, 0].todense()
            if isd:
                G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
            else:
                G = nx.from_numpy_matrix(adj, create_using=nx.Graph())
            embed = LFR(G, w=500)
            embed.train()
            tensors.append(embed.get_embeddings())
    tensors = np.array(tensors)
    np.save(save_path, tensors)
