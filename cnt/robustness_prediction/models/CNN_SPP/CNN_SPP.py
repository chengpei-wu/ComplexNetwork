import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from cnt.robustness_prediction.models.CNN_SPP.SpatialPyramidPooling import SpatialPyramidPooling


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, 512, bias=True))
        self.linears.append(nn.Linear(512, 1024, bias=True))
        self.linears.append(nn.Linear(1024, output_dim, bias=True))

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            if i < 2:
                x = F.relu(layer(x))
            else:
                x = F.hardsigmoid(layer(x))
        return x


class CNN_SPP(nn.Module):
    """
    CNN-SPP module

    References
    -----------
    [1] Chengpei Wu, Yang Lou, Lin Wang, Junli Li, and Guanrong Chen "SPP-CNN: An Efficient Framework for Network Robustness Prediction"
    IEEE Transactions on Circuits and Systems I: Regular Papers doi:10.1109/TCSI.2023.3296602 (2023)

    """

    def __init__(self, output_size=201, levels=[1, 2, 4]):
        super().__init__()
        self.output_size = output_size
        self.levels = levels
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.ReLU()
        )
        self.SpatialPyramidPooling = SpatialPyramidPooling([1, 2, 4])
        _input_size = self.get_mlp_input_size()
        print(_input_size)
        self.MLP = MLP(_input_size, output_size)

    def forward(self, x):
        feature_maps = self.CNN(x)
        pyramid_pooling_fm = self.SpatialPyramidPooling(feature_maps)
        out = self.MLP(pyramid_pooling_fm)
        return out

    def get_mlp_input_size(self):
        return np.sum(np.power(self.levels, 2)) * 256
