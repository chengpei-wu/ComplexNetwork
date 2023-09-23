import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MLP(nn.Module):
    """
    regression module
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, 4096, bias=True))
        self.linears.append(nn.Linear(4096, output_dim, bias=True))

    def forward(self, x):
        for layer in self.linears:
            x = F.relu(layer(x))
        return x


def init_weights(m):
    """
    init weight
    """
    if isinstance(m, nn.Conv2d):
        init.trunc_normal_(m.weight, std=0.01)


class CNN_RP(nn.Module):
    """
    CNN-RP module

    References
    ----------
    [1] Yang Lou, Yaodong He, Lin Wang, and Guanrong Chen "Predicting Network Controllability Robustness: A Convolutional Neural Network Approach"
    IEEE Transactions on Cybernetics vol. 52, no. 5, pp. 4052-4063; doi:10.1109/TCYB.2020.3013251 (2022)

    """

    def __init__(self, input_size=1000, output_size=201):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
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
            nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.CNN.apply(init_weights)
        _input_size = self.get_mlp_input_size()
        self.MLP = MLP(_input_size, output_size)

    def forward(self, x):
        flatten_feature_maps = self.CNN(x)
        out = self.MLP(flatten_feature_maps)
        return out

    def get_mlp_input_size(self):
        _input = torch.rand((1, 1, self.input_size, self.input_size))
        _out = self.CNN(_input)
        return _out.size()[1]
