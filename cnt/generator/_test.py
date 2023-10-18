import matplotlib.pyplot as plt
import networkx as nx

import cnt

# adj = scipy.io.loadmat('./matlab.mat')['ans']
# g0 = nx.from_numpy_array(adj)
# nx.draw(g0, pos=nx.spring_layout(g0), node_size=10, width=1)
# plt.show()

g = cnt.multi_local_world_graph(100, 300, is_directed=True)
print(g)
nx.draw(g, pos=nx.spring_layout(g), node_size=10, width=1)
plt.show()

# degree_distribution = 'power-law'
#
# if degree_distribution == "poisson":
#     degree_sequence = np.random.poisson(lam=4, size=100)
# elif degree_distribution == "uniform":
#     degree_sequence = np.random.uniform(4 - 2, 4 + 2, size=100)
# elif degree_distribution == "normal":
#     degree_sequence = np.random.normal(4, 1, size=100)
# elif degree_distribution == "power-law":
#     degree_sequence = np.random.zipf(2, size=100)
# else:
#     pass
#
# ds = cnt.havel_hakimi_process(list(degree_sequence))
# cnt.havel_hakimi_process(list(ds))
# f1 = nx.is_valid_degree_sequence_havel_hakimi(list(degree_sequence))
# f2 = nx.is_valid_degree_sequence_havel_hakimi(list(ds))
# if not f1 and not f2:
#     print(degree_sequence)
