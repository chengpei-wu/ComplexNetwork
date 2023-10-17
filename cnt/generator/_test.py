import networkx as nx
import numpy as np

import cnt

degree_distribution = 'power-law'

if degree_distribution == "poisson":
    degree_sequence = np.random.poisson(lam=4, size=100)
elif degree_distribution == "uniform":
    degree_sequence = np.random.uniform(4 - 2, 4 + 2, size=100)
elif degree_distribution == "normal":
    degree_sequence = np.random.normal(4, 1, size=100)
elif degree_distribution == "power-law":
    degree_sequence = np.random.zipf(2, size=100)
else:
    pass

ds = cnt.havel_hakimi_process(list(degree_sequence))
cnt.havel_hakimi_process(list(ds))
f1 = nx.is_valid_degree_sequence_havel_hakimi(list(degree_sequence))
f2 = nx.is_valid_degree_sequence_havel_hakimi(list(ds))
if not f1 and not f2:
    print(degree_sequence)
