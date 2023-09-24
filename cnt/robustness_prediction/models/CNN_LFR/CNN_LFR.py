import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    the regression module for LFR-CNN
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, 512, bias=True))
        self.linears.append(nn.Linear(512, 1024, bias=True))
        self.linears.append(nn.Linear(1024, output_dim, bias=True))

    def forward(self, x):
        for layer in self.linears:
            x = F.relu(layer(x))
        return x


class CNN_LFR(nn.Module):
    """
    CNN-LFR module

    References
    -----------
    [1] Yang Lou, Ruizi Wu, Junli Li, Lin Wang, Xiang Li, and Guanrong Chen "A Learning Convolutional Neural Network Approach for Network Robustness Prediction"
    IEEE Transactions on Cybernetics vol. 53, no. 7, pp. 4531-4544; doi:10.1109/TCYB.2022.3207878 (2023)
    """

    def __init__(self, input_size=1000, output_size=201):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten()
        )
        _input_size = self.get_mlp_input_size()
        self.MLP = MLP(_input_size, output_size)

    def forward(self, x):
        flatten_feature_maps = self.CNN(x)
        out = self.MLP(flatten_feature_maps)
        return out

    def get_mlp_input_size(self):
        """
        calculate the input size of MLP
        """
        if self.input_size == 1000:
            _input = torch.rand((1, 1, 125, 160))
        elif self.input_size == 500:
            _input = torch.rand((1, 1, 100, 100))
        else:
            print(f'Input size: {self.input_size} has NOT defined.')
        _out = self.CNN(_input)
        return _out.size()[1]
