import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_pyramid_pool(previous_conv: torch.Tensor, levels: list, mode: str):
    """

    Parameters
    ----------
    previous_conv : the feature maps
    levels : spatial levels
    mode : pooling operator

    Returns
    -------
    the fixed length representation

    """
    num_sample = previous_conv.size(0)
    previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
    for i in range(len(levels)):
        h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
        w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
        w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
        w_pad2 = int(math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
        h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
        h_pad2 = int(math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
        assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
               h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[0])

        padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                             mode='constant', value=0)
        if mode == "max":
            pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
        elif mode == "avg":
            pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
        else:
            raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
        x = pool(padded_input)
        if i == 0:
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)

    return spp


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    """

    def __init__(self, levels, mode="max"):
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        return spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out
