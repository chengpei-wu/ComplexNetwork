import os

import networkx as nx
import torch

from cnt.data.dataset import save_simulated_network_dataset, load_simulated_network_dataset
from cnt.robustness_prediction.models.CNN_RP import CNN_RP
from cnt.robustness_prediction.train_test import train_cnn

if not os.path.exists('./demo_dataset.pkl'):
    save_simulated_network_dataset(
        topology_types=['ba', 'er'],
        is_directed=False,
        is_weighted=False,
        num_instance=500,
        network_size=500,
        average_degree=8,
        save_robustness=['controllability'],
        save_path='./demo_dataset',
        attack='node',
        strategy='degree'
    )

res = load_simulated_network_dataset('./demo_dataset')

# construct dataset
# gnn_dataset = [{'g': dgl.from_networkx(graph), 'label': label} for graph, label in
#                zip(res['graphs'], res['controllability_curves'])]

# init a GIN model
# gnn = GIN(
#     input_dim=1,
#     hidden_dim=16,
#     output_dim=500
# )


# train model
# train_gnn(
#     device=torch.device('cpu'),
#     model=gnn,
#     graphs=gnn_dataset,
#     max_epoch=50,
#     batch_size=4,
#     save_path='./demo_gnn'
# )


# construct dataset
cnn_dataset = [{'g': nx.adjacency_matrix(graph), 'label': label} for graph, label in
               zip(res['graphs'], res['controllability_curves'])]

# init a CNN_RP model
cnn = CNN_RP(
    input_size=500,
    output_size=500
)

# train model
train_cnn(
    device=torch.device('cpu'),
    model=cnn,
    graphs=cnn_dataset,
    max_epoch=50,
    batch_size=4,
    save_path='./demo_cnn'
)
