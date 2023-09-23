import dgl
import numpy as np
import torch


def collate_gnn(samples):
    batch_size = len(samples)
    batch_graphs = [sample['g'] for sample in samples]
    batch_labels = [sample['label'] for sample in samples]
    loop_graphs = [dgl.add_self_loop(graph) for graph in batch_graphs]
    return dgl.batch(loop_graphs), torch.tensor(np.array(batch_labels, dtype=np.float32)).squeeze().float().view(
        batch_size, -1)


def collate_gnn_multi(samples):
    batch_size = len(samples)
    batch_graphs = [sample['g'] for sample in samples]
    batch_labels_pt = [sample['label_pt'] for sample in samples]
    batch_labels_yc = [sample['label_yc'] for sample in samples]
    batch_labels_lc = [sample['label_lc'] for sample in samples]
    batch_labels_cc = [sample['label_cc'] for sample in samples]
    loop_graphs = [dgl.add_self_loop(graph) for graph in batch_graphs]
    return dgl.batch(loop_graphs), torch.tensor(np.array(batch_labels_pt, dtype=np.float32)).squeeze().float().view(
        batch_size, -1), torch.tensor(np.array(batch_labels_yc, dtype=np.float32)).squeeze().float().view(
        batch_size, -1), torch.tensor(np.array(batch_labels_lc, dtype=np.float32)).squeeze().float().view(
        batch_size, -1), torch.tensor(np.array(batch_labels_cc, dtype=np.float32)).squeeze().float().view(
        batch_size, -1)



