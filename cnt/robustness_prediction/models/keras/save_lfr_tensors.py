import networkx as nx
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from CNN_LFR import LFR


def get_lfr_tensors(data_path, w, save_path):
    mat = sio.loadmat(data_path)
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    tensors = []
    temp_adj = networks[0, 0]['adj'][0, 0].todense()
    if np.array_equal(temp_adj.T, temp_adj):
        isd = 0
    else:
        isd = 1
    for i in range(len_net):
        for j in tqdm(range(len_instance), desc=f'saving lfr embeddings: '):
            adj = networks[i, j]['adj'][0, 0].todense()
            if isd:
                G = nx.from_numpy_array(adj, create_using=nx.DiGraph())
            else:
                G = nx.from_numpy_array(adj, create_using=nx.Graph())
            embed = LFR(G, w=w)
            embed.train()
            tensors.append(embed.get_embeddings())
    tensors = np.array(tensors)
    np.save(save_path, tensors)


data_load_path = f'...'
embedding_save_path = f'...'
get_lfr_tensors(data_path=data_load_path, w=1000, save_path=embedding_save_path)
