import networkx as nx

from CompelxNetwork.Robustness.simulated_attack import connectivity_robustness, controllability_robustness, \
    communicability_robustness
from CompelxNetwork.utils.distance_calculation import calculate_EMD


def get_degree_distribution(g):
    return [d[1] for d in g.degree]


class Individual:
    def __init__(self, graph: nx.Graph):
        self.g = graph
        self.R = -1.0
        self.EMD = []
        self.fitness = -100

    def cal_R(self, robustness, attack, strategy):
        if robustness == 'connectivity':
            return connectivity_robustness(self.g, attack, strategy)
        elif robustness == 'controllability':
            return controllability_robustness(self.g, attack, strategy)
        elif robustness == 'communicability':
            return communicability_robustness(self.g, attack, strategy)
        else:
            raise AttributeError(f'{robustness} Not Implemented.')

    def cal_EMD(self, init_graph):
        return calculate_EMD(get_degree_distribution(self.g), get_degree_distribution(init_graph))
