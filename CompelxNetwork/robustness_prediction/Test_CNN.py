import scipy.io as sio
import torch
from parameters import test_params
from torch.utils.data import DataLoader
from utils.utils_cnn import load_labels, load_lfr_embeddings, load_network_cnn, load_var_network, collate_cnn
from utils.utils_train_test import predict

from models.CNN_LFR import CNN_LFR
from models.CNN_SPP import CNN_SPP

exp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for cnn_model in ['cnn_lfr', 'cnn_spp', 'cnn_rp']:
    for atk in ['ndeg']:
        for isd in [0]:
            for label in ['yc']:
                test_params["cnn_model"] = cnn_model
                test_params["atk"] = atk
                test_params["isd"] = isd
                test_params["label"] = label
                load_model_path = f'./checkpoints/{test_params["cnn_model"]}/real_{test_params["atk"]}_isd{test_params["isd"]}_{test_params["label"]}.hdf5'
                load_data_path = f'./data/test/testing_real_networks_{test_params["atk"]}_isd{test_params["isd"]}.mat'
                save_pred_path = f'./prediction/{test_params["cnn_model"]}/real_{test_params["atk"]}_isd{test_params["isd"]}_{test_params["label"]}.mat'
                if test_params['label'] in ['pt', 'cc']:
                    output_dim = 1
                else:
                    output_dim = 201

                assert test_params["label"] != 'all'

                model = torch.load(load_model_path)
                if test_params["cnn_model"] == 'cnn_lfr':
                    embedding_load_path = f'./data/lfr_embeddings/test/testing_real_networks_{test_params["atk"]}_isd{test_params["isd"]}.npy'
                    load_data_path = embedding_load_path
                if isinstance(model, CNN_LFR):
                    x = load_lfr_embeddings(load_data_path, w=500)
                    y = load_labels(load_data_path, label)
                    graphs = list(zip(x, y))
                elif isinstance(model, CNN_SPP):
                    graphs = load_network_cnn(load_data_path, label)
                else:
                    graphs = load_var_network(load_data_path, label, fixed_size=500, sampling='random')
                test_loader = DataLoader(
                    graphs,
                    batch_size=1,
                    collate_fn=collate_cnn,
                )

            prediction = predict(
                test_loader, exp_device, model
            )
            sio.savemat(save_pred_path, prediction)
            print(f'\nprediction results saved in: {save_pred_path}')
