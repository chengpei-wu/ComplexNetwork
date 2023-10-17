import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import *


class MLP(nn.Module):
    """
    Construct two-layer MLP-type aggregator for GIN model
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    """
    See How powerful are graph neural networks. Xu et.al.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            AvgPooling()
        )  # change to sum readout

    def forward(self, g):
        # list of hidden representation at each layer (including the input layer)
        h = g.in_degrees().view(-1, 1).float()
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph Avg. pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.linear_prediction[i](pooled_h)
        score_over_layer = score_over_layer / len(hidden_rep)
        activated_out = F.hardsigmoid(score_over_layer)
        return activated_out


class Multi_GIN(nn.Module):
    """
    multi-task GIN for robustness prediction
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction_pt = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction_pt.append(nn.Linear(input_dim, 1))
            else:
                self.linear_prediction_pt.append(nn.Linear(hidden_dim, 1))

        self.linear_prediction_yc = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction_yc.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction_yc.append(nn.Linear(hidden_dim, output_dim))

        self.linear_prediction_lc = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction_lc.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction_lc.append(nn.Linear(hidden_dim, output_dim))

        self.linear_prediction_cc = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction_cc.append(nn.Linear(input_dim, 1))
            else:
                self.linear_prediction_cc.append(nn.Linear(hidden_dim, 1))

        self.drop = nn.Dropout(0.5)
        self.pool = (
            AvgPooling()
        )  # change to sum readout

    def forward(self, g):
        # list of hidden representation at each layer (including the input layer)
        h = g.in_degrees().view(-1, 1).float()
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer_pt, score_over_layer_yc, score_over_layer_lc, score_over_layer_cc = 0, 0, 0, 0
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer_pt += self.linear_prediction_pt[i](pooled_h)
            score_over_layer_yc += self.linear_prediction_yc[i](pooled_h)
            score_over_layer_lc += self.linear_prediction_lc[i](pooled_h)
            score_over_layer_cc += self.linear_prediction_cc[i](pooled_h)

        score_over_layer_pt = score_over_layer_pt / len(hidden_rep)
        score_over_layer_yc = score_over_layer_yc / len(hidden_rep)
        score_over_layer_lc = score_over_layer_lc / len(hidden_rep)
        score_over_layer_cc = score_over_layer_cc / len(hidden_rep)
        activated_out_pt = F.hardsigmoid(score_over_layer_pt)
        activated_out_yc = F.hardsigmoid(score_over_layer_yc)
        activated_out_lc = F.hardsigmoid(score_over_layer_lc)
        activated_out_cc = F.hardsigmoid(score_over_layer_cc)
        return activated_out_pt, activated_out_yc, activated_out_lc, activated_out_cc
