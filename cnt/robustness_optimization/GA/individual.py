from typing import Union

import networkx as nx
import numpy as np

from cnt.robustness.simulated_attack import connectivity_robustness, controllability_robustness, \
    communicability_robustness
from cnt.utils.distance_calculation import calculate_EMD


def get_degree_distribution(graph: Union[nx.Graph, nx.DiGraph]) -> list:
    """
    get node degrees of a graph

    Parameters
    ----------
    graph : inoput graph

    Returns
    -------

    a list of degrees

    """
    return [d[1] for d in nx.degree(graph)]


class Individual:
    def __init__(self, graph: Union[nx.Graph, nx.DiGraph]):
        self.g = graph
        self.R = -1.0
        self.EMD = []
        self.fitness = -100

    def cal_R(self, robustness: str, attack: str, strategy: str) -> np.ndarray:
        """
        calculate network robustness

        Parameters
        ----------
        robustness : robustness type
        attack : nodes or edges attack
        strategy : strategy of choosing targets under attacks

        Returns
        -------
        robustness value (R, average of robustness curve)

        """
        if robustness == 'connectivity':
            return connectivity_robustness(self.g, attack, strategy)[1]
        elif robustness == 'controllability':
            return controllability_robustness(self.g, attack, strategy)[1]
        elif robustness == 'communicability':
            return communicability_robustness(self.g, attack, strategy)[1]
        else:
            raise AttributeError(f'{robustness} Not Implemented.')

    def cal_EMD(self, init_graph: Union[nx.Graph, nx.DiGraph]) -> float:
        """
        calculate the Wasserstein Distance of degree distributions of two graphs.

        Parameters
        ----------
        init_graph : the original input graph (to be optimized)

        Returns
        -------
        the EMD (Wasserstein) distance

        """
        return calculate_EMD(get_degree_distribution(self.g), get_degree_distribution(init_graph))
