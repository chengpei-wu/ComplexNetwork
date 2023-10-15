import matplotlib.pyplot as plt

from cnt.generator.generator import *

g = generic_scale_free_graph(1000, 4000)
ds = [d[1] for d in g.degree()]
plt.hist(ds, bins=50)
plt.show()

g = barabasi_albert_graph(1000, 4000)
ds = [d[1] for d in g.degree()]
plt.hist(ds, bins=50)
plt.show()
