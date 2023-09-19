from typing import Union

import networkx as nx
import numpy as np

from CompelxNetwork.Robustness.simulated_attack import connectivity_robustness, controllability_robustness, \
    communicability_robustness
from CompelxNetwork.utils.distance_calculation import calculate_EMD


def get_degree_distribution(graph: Union[nx.Graph, nx.DiGraph]) -> list:
    return [d[1] for d in nx.degree(graph)]


class Individual:
    def __init__(self, graph: Union[nx.Graph, nx.DiGraph]):
        self.g = graph
        self.R = -1.0
        self.EMD = []
        self.fitness = -100

    def cal_R(self, robustness: str, attack: str, strategy: str) -> np.ndarray:
        if robustness == 'connectivity':
            return connectivity_robustness(self.g, attack, strategy)[1]
        elif robustness == 'controllability':
            return controllability_robustness(self.g, attack, strategy)[1]
        elif robustness == 'communicability':
            return communicability_robustness(self.g, attack, strategy)[1]
        else:
            raise AttributeError(f'{robustness} Not Implemented.')

    def cal_EMD(self, init_graph: Union[nx.Graph, nx.DiGraph]) -> int:
        return calculate_EMD(get_degree_distribution(self.g), get_degree_distribution(init_graph))
