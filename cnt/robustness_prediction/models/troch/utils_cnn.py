import numpy as np
import torch


def collate_cnn(samples):
    batch_size = len(samples)
    batch_graphs = [sample['g'].todense() for sample in samples]
    batch_labels = [sample['label'] for sample in samples]
    return torch.tensor(np.array(batch_graphs, dtype=np.float32)).view(
        batch_size, 1, len(batch_graphs[0]), len(batch_graphs[0]), ), torch.tensor(
        np.array(batch_labels, dtype=np.float32)).squeeze().float().view(
        batch_size, -1)
