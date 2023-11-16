import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw
from scipy import stats


def _power_law_numbers(num_points, sum_total, exponent):
    pl = powerlaw.Power_Law(parameters=[exponent])
    numbers = np.round(pl.generate_random(num_points)).astype(int)
    current_sum = np.sum(numbers)
    numbers = numbers * (sum_total / current_sum)
    numbers = np.round(numbers).astype(int)
    numbers[numbers <= 0] = 1
    return list(numbers)


for i in range(10000):
    degrees = _power_law_numbers(100, 200 * 2, 2.5)
    g = nx.expected_degree_graph(degrees)
    data = [d for _, d in g.degree()]
    print(data)
