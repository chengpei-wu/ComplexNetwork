from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from cnt.robustness_prediction.models.CNN_SPP import CNN_SPP
from cnt.robustness_prediction.utils.utils_cnn import collate_cnn
from cnt.robustness_prediction.utils.utils_gnn import collate_gnn, collate_gnn_multi
from cnt.utils.tool_functions import print_progress


def train_cnn(
        device: torch.device,
        model: torch.nn.Module,
        graphs: list,
        max_epoch: int,
        batch_size: int,
        save_path: str
):
    """
    training for a CNN model
    Parameters
    ----------
    device : device: GPU or CPU
    model : the trained model
    graphs : the trained graphs with labels: [(graph_0, label_0), ..., (graph_n, label_n)]
    max_epoch : the maximum number of training epochs
    batch_size : the batch size of training setting
    save_path : the save model path

    Returns
    -------
    the training losses per steps: [[mae, val_mae], ...]

    """
    graphs = shuffle(graphs)
    # using 9/10 of data for training, the rest for validation
    train_data_number = round(len(graphs) * 0.9)
    train_loader = DataLoader(
        graphs[:train_data_number],
        batch_size=batch_size,
        collate_fn=collate_cnn,
    )
    val_loader = DataLoader(
        graphs[train_data_number:],
        batch_size=1,
        collate_fn=collate_cnn,
    )
    # loss function, optimizer and scheduler
    loss_fcn = nn.MSELoss()
    # loss_fcn = peak_loss()
    if isinstance(model, CNN_SPP):
        optimizer = optim.SGD(model.parameters(), lr=0.1)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    model.to(device)
    # training loop
    loss_history = []
    all_val_mae = []
    for epoch in range(max_epoch):
        model.train()
        total_loss = 0
        for batch, (batched_graph, robustness) in enumerate(train_loader):
            print_progress(batch, train_data_number // batch_size, prefix=f'Epoch {epoch}: ')
            batched_graph = batched_graph.to(device)
            robustness = robustness.to(device)
            logits = model(batched_graph)
            loss = loss_fcn(logits, robustness)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # scheduler.step()
        mse, mae = evaluate(train_loader, device, model)
        val_mse, val_mae = evaluate(val_loader, device, model)
        loss_history.append([mae, val_mae])
        print(loss_history)
        all_val_mae.append(val_mae)
        print()
        print(
            f'Epoch {epoch} | Loss {total_loss / (batch + 1):8.2f} | mse. {mse:8.3f} | val_mse. {val_mse:8.3f}| mae. {mae:8.3f} | '
            f'val_mae. {val_mae:8.7f} ')
        if val_mae <= np.min(all_val_mae):
            print(f'val_mae -> {val_mae}, save model.')
            # save the best model guided by val_mae
            torch.save(model, save_path)
        # if val_mae <= 0.005:
        #     print(f'Epoch {epoch}, valid mae is less than 0.005, stop training.')
    return loss_history


def train_gnn(
        device: torch.device,
        model: torch.nn.Module,
        graphs: list,
        max_epoch: int,
        batch_size: int,
        save_path: str
):
    """
    training for a GNN model
    Parameters
    ----------
    device : device: GPU or CPU
    model : the trained model
    graphs : the trained graphs with labels: [(graph_0, label_0), ..., (graph_n, label_n)]
    max_epoch : the maximum number of training epochs
    batch_size : the batch size of training setting
    save_path : the save model path

    Returns
    -------
    the training losses per steps: [[mae, val_mae], ...]

    """
    graphs = shuffle(graphs)
    # using 9/10 of data for training, the rest for validation
    train_data_number = round(len(graphs) * 0.9)
    train_loader = DataLoader(
        graphs[:train_data_number],
        batch_size=batch_size,
        collate_fn=collate_gnn,
    )
    val_loader = DataLoader(
        graphs[train_data_number:],
        batch_size=256,
        collate_fn=collate_gnn,
    )
    # loss function, optimizer and scheduler
    loss_fcn = nn.MSELoss()
    # loss_fcn = peak_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    model.to(device)
    # training loop
    loss_history = []
    all_val_mae = []
    for epoch in range(max_epoch):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            print_progress(batch, train_data_number // batch_size, prefix=f'Epoch {epoch}: ')
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            logits = model(batched_graph)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # scheduler.step()
        mse, mae = evaluate(train_loader, device, model)
        val_mse, val_mae = evaluate(val_loader, device, model)
        loss_history.append([mae, val_mae])
        print(loss_history)
        all_val_mae.append(val_mae)
        print()
        print(
            f'Epoch {epoch} | Loss {total_loss / (batch + 1):8.2f} | mse. {mse:8.3f} | val_mse. {val_mse:8.3f}| mae. {mae:8.3f} | '
            f'val_mae. {val_mae:8.7f} ')
        if val_mae <= np.min(all_val_mae):
            print(f'val_mae -> {val_mae}, save model.')
            # save the best model guided by val_mae
            torch.save(model, save_path)
        # if val_mae <= 0.005:
        #     print(f'Epoch {epoch}, valid mae is less than 0.005, stop training.')
    return loss_history


# training GNN model, with multi-task learning
def train_multi_gnn(
        device: torch.device,
        model: torch.nn.Module,
        graphs: list,
        max_epoch: int,
        batch_size: int,
        save_path: str
):
    """
    training for a multi-task GNN model
    Parameters
    ----------
    device : device: GPU or CPU
    model : the trained model
    graphs : the trained graphs with labels: [(graph_0, label_0), ..., (graph_n, label_n)]
    max_epoch : the maximum number of training epochs
    batch_size : the batch size of training setting
    save_path : the save model path

    Returns
    -------
    the training losses per steps: [[mae, val_mae], ...]

    """
    graphs = shuffle(graphs)
    train_data_number = round(len(graphs) * 0.9)
    train_loader = DataLoader(
        graphs[:train_data_number],
        batch_size=batch_size,
        collate_fn=collate_gnn_multi,
    )
    val_loader = DataLoader(
        graphs[train_data_number:],
        batch_size=256,
        collate_fn=collate_gnn_multi,
    )
    # loss function, optimizer and scheduler
    loss_fcn = nn.MSELoss()
    # loss_fcn = peak_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    model.to(device)
    # training loop
    all_val_mae_pt, all_val_mae_yc, all_val_mae_lc, all_val_mae_cc = [], [], [], []
    loss_history = []
    for epoch in range(max_epoch):
        model.train()
        L1, L2, L3, L4 = [], [], [], []
        for batch, (batched_graph, labels_pt, labels_yc, labels_lc, labels_cc) in enumerate(train_loader):
            print_progress(batch, train_data_number // batch_size, prefix=f'Epoch {epoch}: ')
            batched_graph = batched_graph.to(device)
            labels_pt = labels_pt.to(device)
            labels_yc = labels_yc.to(device)
            labels_lc = labels_lc.to(device)
            labels_cc = labels_cc.to(device)
            logits_pt, logits_yc, logits_lc, logits_cc = model(batched_graph)
            loss_pt = loss_fcn(logits_pt, labels_pt)
            loss_yc = loss_fcn(logits_yc, labels_yc)
            loss_lc = loss_fcn(logits_lc, labels_lc)
            loss_cc = loss_fcn(logits_cc, labels_cc)
            optimizer.zero_grad()
            L1.append(loss_pt.item())
            L2.append(loss_yc.item())
            L3.append(loss_lc.item())
            L4.append(loss_cc.item())
            w1, w2, w3, w4 = np.var(L1) + 1e-8, np.var(L2) + 1e-8 + 1e-8, np.var(L3) + 1e-8, np.var(L4) + 1e-8
            ws = w1 + w2 + w3 + w4
            w1, w2, w3, w4 = w1 / ws, w2 / ws, w3 / ws, w4 / ws
            # print(w1, w2, w3, w4)
            all_loss = loss_pt / w1 + loss_yc / w2 + loss_lc / w3 + loss_cc / w4
            # print(all_loss)
            all_loss.backward()
            optimizer.step()
        # scheduler.step()
        mse_pt, mae_pt, mse_yc, mae_yc, mse_lc, mae_lc, mse_cc, mae_cc = evaluate_multi(train_loader, device, model)
        val_mse_pt, val_mae_pt, val_mse_yc, val_mae_yc, val_mse_lc, val_mae_lc, val_mse_cc, val_mae_cc = evaluate_multi(
            val_loader, device,
            model)
        pre_val_mae_pt = all_val_mae_pt[-1] if len(all_val_mae_pt) > 0 else 'inf'
        pre_val_mae_yc = all_val_mae_yc[-1] if len(all_val_mae_yc) > 0 else 'inf'
        pre_val_mae_lc = all_val_mae_lc[-1] if len(all_val_mae_lc) > 0 else 'inf'
        pre_val_mae_cc = all_val_mae_cc[-1] if len(all_val_mae_cc) > 0 else 'inf'
        all_val_mae_pt.append(val_mae_pt)
        all_val_mae_yc.append(val_mae_yc)
        all_val_mae_lc.append(val_mae_lc)
        all_val_mae_cc.append(val_mae_cc)
        loss_history.append([mae_yc, val_mae_yc, mae_lc, val_mae_lc])
        print()
        print(
            f'Epoch {epoch} | mae_pt. {mae_pt:5.3f} | val_mae_pt. {val_mae_pt:5.3f}| mae_yc. {mae_yc:5.3f} | '
            f'val_mae_yc. {val_mae_yc:5.3f} | mae_lc. {mae_lc:5.3f} | val_mae_lc. {val_mae_lc:5.3f}| mae_cc. {mae_cc:5.3f} | val_mae_cc. {val_mae_cc:5.3f}')
        if val_mae_pt <= np.min(all_val_mae_pt) or val_mae_yc <= np.min(all_val_mae_yc) or val_mae_lc <= np.min(
                all_val_mae_lc) or val_mae_cc <= np.min(all_val_mae_cc):
            print(f'val_mae_pt: {pre_val_mae_pt} ---> {val_mae_pt}.')
            print(f'val_mae_yc: {pre_val_mae_yc} ---> {val_mae_yc}.')
            print(f'val_mae_lc: {pre_val_mae_lc} ---> {val_mae_lc}.')
            print(f'val_mae_cc: {pre_val_mae_cc} ---> {val_mae_cc}.')
            print('save model!')
            torch.save(model, save_path)
        # if val_mae_pt <= 0.004 and val_mae_yc <= 0.005 and val_mae_lc <= 0.005 and val_mae_cc <= 0.004:
        #     print(f'Epoch {epoch}, valid mae is less enough, stop training.')
    return np.array(loss_history)


# evaluate validation data for GNN models
def evaluate(data_loader, device, model):
    model.eval()
    mse, mae = [], []
    for batched_graph, labels in data_loader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        logits = model(batched_graph)
        mse.append(mean_squared_error(labels.cpu().detach().numpy(), logits.cpu().detach().numpy()))
        mae.append(mean_absolute_error(labels.cpu().detach().numpy(), logits.cpu().detach().numpy()))
    return np.mean(mse), np.mean(mae)


# evaluate validation data for GNN multi-task models
def evaluate_multi(data_loader, device, model):
    model.eval()
    mse_pt, mae_pt, mse_yc, mse_cc, mae_yc, mse_lc, mae_lc, mae_cc = [], [], [], [], [], [], [], []
    for batched_graph, labels_pt, labels_yc, labels_lc, labels_cc in data_loader:
        batched_graph = batched_graph.to(device)
        labels_pt = labels_pt.to(device)
        labels_yc = labels_yc.to(device)
        labels_lc = labels_lc.to(device)
        labels_cc = labels_cc.to(device)
        logits_pt, logits_yc, logits_lc, logits_cc = model(batched_graph)
        mse_pt.append(mean_squared_error(labels_pt.cpu().detach().numpy(), logits_pt.cpu().detach().numpy()))
        mae_pt.append(mean_absolute_error(labels_pt.cpu().detach().numpy(), logits_pt.cpu().detach().numpy()))

        mse_yc.append(mean_squared_error(labels_yc.cpu().detach().numpy(), logits_yc.cpu().detach().numpy()))
        mae_yc.append(mean_absolute_error(labels_yc.cpu().detach().numpy(), logits_yc.cpu().detach().numpy()))

        mse_lc.append(mean_squared_error(labels_lc.cpu().detach().numpy(), logits_lc.cpu().detach().numpy()))
        mae_lc.append(mean_absolute_error(labels_lc.cpu().detach().numpy(), logits_lc.cpu().detach().numpy()))

        mse_cc.append(mean_squared_error(labels_cc.cpu().detach().numpy(), logits_cc.cpu().detach().numpy()))
        mae_cc.append(mean_absolute_error(labels_cc.cpu().detach().numpy(), logits_cc.cpu().detach().numpy()))
    return np.mean(mse_pt), np.mean(mae_pt), np.mean(mse_yc), np.mean(mae_yc), np.mean(mse_lc), np.mean(
        mae_lc), np.mean(mse_cc), np.mean(mae_cc)


# predict testing data
def predict(data_loader, robustness, device, model):
    model.eval()
    sim = []
    pred = []
    tic = time()
    for batched_graph, labels in data_loader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        logits = model(batched_graph)
        temp_logits = logits.cpu().detach().numpy()
        temp_labels = labels.cpu().detach().numpy()
        if len(pred) == 0:
            sim = temp_labels
            pred = temp_logits
        else:
            sim = np.concatenate((sim, temp_labels))
            pred = np.concatenate((pred, temp_logits))
    toc = time() - tic
    pred = np.array(pred).squeeze()
    sim = np.array(sim).squeeze()
    if robustness == 'pt':
        return calculate_horizontal_accuracy(pred, sim, toc / 900)
    elif robustness == 'cc':
        return calculate_vertical_accuracy(pred, sim, toc / 900)
    else:
        mae = np.mean(np.abs(sim - pred), axis=1)
    if robustness == 'yc':
        prediction = {
            'sim_yc': sim,
            'pred_yc': pred,
            'time': toc / 900,
            'mae_yc': mae
        }
    if robustness == 'lc':
        prediction = {
            'sim_lc': sim,
            'pred_lc': pred,
            'time': toc / 900,
            'mae_lc': mae
        }
    return prediction


# predict testing data for multi-task GNNs
def predict_multi_gnn(data_loader, device, model):
    model.eval()
    sim_pt, sim_yc, sim_lc, sim_cc = [], [], [], []
    pred_pt, pred_yc, pred_lc, pred_cc = [], [], [], []
    tic = time()
    for batched_graph, labels_pt, labels_yc, labels_lc, labels_cc in data_loader:
        batched_graph = batched_graph.to(device)
        labels_pt = labels_pt.to(device)
        labels_yc = labels_yc.to(device)
        labels_lc = labels_lc.to(device)
        labels_cc = labels_cc.to(device)
        logits_pt, logits_yc, logits_lc, logits_cc = model(batched_graph)
        temp_logits_pt = logits_pt.cpu().detach().numpy()
        temp_logits_yc = logits_yc.cpu().detach().numpy()
        temp_logits_lc = logits_lc.cpu().detach().numpy()
        temp_logits_cc = logits_cc.cpu().detach().numpy()
        temp_labels_pt = labels_pt.cpu().detach().numpy()
        temp_labels_yc = labels_yc.cpu().detach().numpy()
        temp_labels_lc = labels_lc.cpu().detach().numpy()
        temp_labels_cc = labels_cc.cpu().detach().numpy()
        if len(pred_pt) == 0:
            sim_pt = temp_labels_pt
            sim_yc = temp_labels_yc
            sim_lc = temp_labels_lc
            sim_cc = temp_labels_cc
            pred_pt = temp_logits_pt
            pred_yc = temp_logits_yc
            pred_lc = temp_logits_lc
            pred_cc = temp_logits_cc
        else:
            sim_pt = np.concatenate((sim_pt, temp_labels_pt))
            sim_yc = np.concatenate((sim_yc, temp_labels_yc))
            sim_lc = np.concatenate((sim_lc, temp_labels_lc))
            sim_cc = np.concatenate((sim_cc, temp_labels_cc))
            pred_pt = np.concatenate((pred_pt, temp_logits_pt))
            pred_yc = np.concatenate((pred_yc, temp_logits_yc))
            pred_lc = np.concatenate((pred_lc, temp_logits_lc))
            pred_cc = np.concatenate((pred_cc, temp_logits_cc))
    toc = time() - tic
    pred_pt = np.round(np.round(np.array(pred_pt).squeeze(), 3) * 200) / 2
    pred_yc = np.array(pred_yc).squeeze()
    pred_lc = np.array(pred_lc).squeeze()
    pred_cc = np.round(np.array(pred_cc).squeeze(), 3) * 100
    sim_pt = np.round(np.round(np.array(sim_pt).squeeze(), 3) * 200) / 2
    sim_yc = np.array(sim_yc).squeeze()
    sim_lc = np.array(sim_lc).squeeze()
    sim_cc = np.round(np.array(sim_cc).squeeze(), 3) * 100
    mae_yc = np.mean(np.abs(sim_yc - pred_yc), axis=1)
    mae_lc = np.mean(np.abs(sim_lc - pred_lc), axis=1)

    peak_acc1, peak_acc2 = [], []
    for std in np.arange(0, 5, 0.5):
        peak_acc1.append(np.sum(np.abs(sim_pt - pred_pt) <= std) / len(pred_pt))
        peak_acc2.append(np.sum(np.abs(sim_cc - pred_cc) <= std) / len(pred_cc))
    prediction = {
        'sim_yc': sim_yc,
        'sim_lc': sim_lc,
        'sim_t': sim_pt,
        'sim_m': sim_cc,
        'pred_yc': pred_yc,
        'pred_lc': pred_lc,
        'pred_t': pred_pt,
        'pred_m': pred_cc,
        'mae_yc': mae_yc,
        'mae_lc': mae_lc,
        'accuracy_t': peak_acc1,
        'accuracy_m': peak_acc2,
        'time': toc / 900
    }
    return prediction


def calculate_vertical_accuracy(pred, sim, time, max_std=5):
    prediction = {
        'pred_m': np.round(pred, 3) * 100,
        'sim_m': np.round(sim, 3) * 100,
        'time': time
    }
    peak_acc = []
    for std in np.arange(0, max_std, 0.5):
        peak_acc.append(np.sum(np.abs(prediction['sim_m'] - prediction['pred_m']) <= std) / len(prediction['sim_m']))
    prediction['accuracy_m'] = peak_acc
    return prediction


def calculate_horizontal_accuracy(pred, sim, time, max_std=5):
    prediction = {
        'pred_t': np.round(pred, 3) * 100,
        'sim_t': np.round(sim, 3) * 100,
        'time': time
    }
    peak_acc = []
    for std in np.arange(0, max_std, 0.5):
        peak_acc.append(np.sum(np.abs(prediction['sim_t'] - prediction['pred_t']) <= std) / len(prediction['sim_t']))
    prediction['accuracy_t'] = peak_acc
    return prediction
