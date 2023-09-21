import dgl
import numpy as np
import scipy.io as sio
import torch

from utils.utils_tool_function import *


# load networks from .mat files
# transform adjacency matrix to dgl.Graph
def load_network_gnn(path, robustness):
    mat = sio.loadmat(path)
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    graphs = []
    for i in range(len_net):
        for j in range(len_instance):
            print_progress(i * len_instance + j + 1, len_net * len_instance, prefix='loading_networks:', )
            adj = networks[i, j]['adj'][0][0]
            g = dgl.from_scipy(adj)
            if robustness == 'lc':
                label = networks[i, j]['lc'][0][0]
            if robustness == 'yc':
                label = networks[i, j]['yc'][0][0]
            if robustness == 'pt':
                pt = networks[i, j]['pt'][0][0].squeeze()
                peak = (np.mean(np.where(pt == np.amax(pt))) / adj.shape[0])
                label = peak
            if robustness == 'cc':
                pt = networks[i, j]['pt'][0][0].squeeze()
                cc_peak = np.amax(pt)
                label = cc_peak / adj.shape[0]
            if robustness == 'all':
                label_lc = networks[i, j]['lc'][0][0]
                label_yc = networks[i, j]['yc'][0][0]
                pt = networks[i, j]['pt'][0][0].squeeze()
                peak = (np.mean(np.where(pt == np.amax(pt))) / adj.shape[0])
                cc_peak = np.amax(pt)
                label_pt = peak
                label_cc = cc_peak / adj.shape[0]
            if robustness == 'all':
                graphs.append(
                    {'g': g, 'label_lc': label_lc, 'label_yc': label_yc, 'label_pt': label_pt, 'label_cc': label_cc})
            else:
                graphs.append({'g': g, 'label': label})
    print()
    return graphs


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


def calculate_param_number(model):
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
