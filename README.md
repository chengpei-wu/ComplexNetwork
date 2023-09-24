# complex-network-tools

complex-network-tools is a Python package for generating, learning, and analysis of complex networks.

# Examples

##### Synthetic network generating

```python
from cnt.generator import erdos_renyi_graph

# generate a Erdos Renyi(ER) random graph
G = erdos_renyi_graph(num_nodes=100, num_edges=400, is_directed=False, is_weighted=False)

# generate a Barabasi Albert(BA) scale-free graph
G = barabasi_albert_graph(num_nodes=100, num_edges=400, is_directed=False, is_weighted=False) 
```

##### Network attack

```python
from cnt.robustness.network_attack import network_attack

# get attack sequence of nodes
G = erdos_renyi_graph(num_nodes=100, num_edges=400, is_directed=False, is_weighted=False)
# node-removal based network attacks, use the targeted-degree based node-removal strategy
attack_sequence = network_attack(graph=G, attack='node', strategy='degree')
```

##### Spectral measure

```python
from cnt.spectral_measure.robustness_spectral_measure import spectral_gap

# calculate spectral gap for a graph
G = erdos_renyi_graph(num_nodes=100, num_edges=400, is_directed=False, is_weighted=False)
spectral_gap(G)
```

# Install

```shell
pip install complex-network-tools
```
