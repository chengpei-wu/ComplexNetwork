# complex-network-tools

complex-network-tools is a Python package for generating, learning, and analysis of complex networks.

# Install

```shell
pip install complex-network-tools
```

# Examples

### Synthetic network generating

```python
import cnt

# generate a Erdos Renyi(ER) random graph
er_graph = cnt.erdos_renyi_graph(num_nodes=100, num_edges=400, is_directed=False, is_weighted=False)

# generate a Barabasi Albert(BA) scale-free graph
ba_graph = cnt.barabasi_albert_graph(num_nodes=100, num_edges=400, is_directed=False, is_weighted=False)

```

### Network attack

```python
import cnt

# get attack sequence of nodes
G = cnt.erdos_renyi_graph(num_nodes=100, num_edges=400, is_directed=False, is_weighted=False)
# node-removal based network attacks, use the targeted-degree based node-removal strategy
attack_node_sequence = cnt.network_attack(graph=G, attack='node', strategy='degree')
```

### Spectral measure

```python
import cnt

G = cnt.erdos_renyi_graph(num_nodes=100, num_edges=400, is_directed=False, is_weighted=False)

# calculate spectral gap
spectral_gap = cnt.spectral_gap(G)

# calculate spectral radius
spectral_radius = cnt.spectral_radius(G)

# calculate natural_connectivity
natural_connectivity = cnt.natural_connectivity(G)

# calculate algebraic_connectivity
algebraic_connectivity = cnt.algebraic_connectivity(G)
```

### Network dataset of network robustness simulation

```python
import cnt

# generating and saving dataset of networks with their robustness
dataset = cnt.save_simulated_network_dataset(
    topology_types=['er', 'ba'],
    is_directed=False,
    is_weighted=False,
    num_instance=100,
    network_size=(300, 500),
    average_degree=(5, 10),
    save_robustness=['connectivity', 'controllability', 'communicability'],
    attack='node',
    strategy='degree',
    save_path='./dataset/file_name'
)

# loading dataset
dataset = cnt.load_simulated_network_dataset(load_path='./dataset/file_name')

# get graphs
graphs = dataset['graphs']

# get labels
connectivity_robustness = dataset['connectivity_curves']

```

### Training deep models for network robustness prediction

```python
import networkx as nx
import cnt

# loading dataset
dataset = cnt.load_simulated_network_dataset(load_path='./dataset/file_name')

# get graphs
graphs = dataset['graphs']

# get labels
connectivity_robustness = dataset['connectivity_curves']

# init CNN-RP
# cnn_rp = cnt.keras_models.CNN_RP(
#     input_size=500
# )

# init CNN-SPP
cnn_spp = cnt.keras_models.CNN_SPP()

# training
cnn_spp.fit(
    x=[nx.adj_matrix(g) for g in graphs],
    y=[cnt.uniform_sampling(curve, 5) for curve in connectivity_robustness],
    model_path='./checkpiont/model'
)

```

### Network robustness optimization using a simple GA

```python
import cnt

# create initial graph
init_graph = cnt.erdos_renyi_graph(num_nodes=100, num_edges=400, is_directed=False, is_weighted=False)

# create initial population
Pop = cnt.GA.Population(init_graph=init_graph, init_size=10, max_size=25)

# optimization
all_best_ind = None

for gen in range(100):
    Pop.crossover()
    Pop.mutate()
    Pop.selection()
    best_ind = Pop.find_best()
    if gen == 0:
        all_best_ind = best_ind
    else:
        if best_ind.R > all_best_ind.R:
            all_best_ind = best_ind

```

### useful functions

```python
import networkx as nx
import cnt

G = cnt.erdos_renyi_graph(100, 400)

# sort adjacency matrix
cnt.adj_sort(adj=nx.adj_matrix(G))

# shuffle adjacency matrix
cnt.adj_shuffle(adj=nx.adj_matrix(G))

# random sampling for adjacency matrix
cnt.random_sampling(adj=nx.adj_matrix(G), fixed_size=500)

# Bilinear interpolation for adjacency matrix
cnt.bi_linear_sampling(adj=nx.adj_matrix(G), fixed_size=500)

# nodes noises
cnt.missing_nodes(adj=nx.adj_matrix(G), strategy='random', rate=0.1)
```