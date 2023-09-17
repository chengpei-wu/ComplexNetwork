import networkx as nx

from utils import calculate_robustness, get_degree_distribution, calculate_EMD


class Individual:
    def __init__(self, graph: nx.Graph):
        self.g = graph
        self.R = -1.0
        self.EMD = []
        self.fitness = -100

    def cal_R(self):
        return calculate_robustness(self.g)

    def cal_EMD(self, o_g):
        return calculate_EMD(get_degree_distribution(self.g), get_degree_distribution(o_g))
