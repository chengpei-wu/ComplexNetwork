import networkx as nx
import numpy as np

from cnt.robustness_prediction.models.CNN_LFR.receptive_field import ReceptiveField


class LFR:
    """
    the learning feature representation (Patchy-SAN) module

    References
    -----------
    [1] Mathias Niepert, Mohamed Ahmed, and Konstantin Kutzkov, "Learning Convolutional Neural Networks for Graphs" ICML

    """

    def __init__(self, G, w, num_attr=2, s=1, k=10, l='betweenness'):
        self.G = G
        self.s = s
        self.w = w
        self.k = k
        self.l = l
        self.attribute_name = 'node_attributes'
        self.embedding = []
        self.num_attr = num_attr
        self.init_attr()

    def init_attr(self):
        # assign attrs to every node
        feats_deg = self.G.degree()
        feats_cc = nx.clustering(self.G)
        for k in range(self.G.number_of_nodes()):
            attr = [feats_deg[k], feats_cc[k]]
            self.G.nodes[k]['node_attributes'] = attr

    def train(self):
        rf = ReceptiveField(
            self.G,
            w=self.w,
            k=self.k,
            s=self.s,
            l=self.l,
            attribute_name=self.attribute_name,
            num_attr=self.num_attr
        )

        receptive_fields = rf.make_all_receptive_fields()
        rf_tensor = np.array(receptive_fields).flatten().reshape(self.w * self.k, self.num_attr)
        self.embedding = rf_tensor

    def get_embeddings(self):
        return self.embedding
