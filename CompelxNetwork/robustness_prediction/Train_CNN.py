import torch

from models.CNN_LFR import CNN_LFR
from models.CNN_RP import CNN_RP
from models.CNN_SPP import CNN_SPP
from parameters import train_params
from utils.utils_train_test import train_cnn

exp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for cnn_model in ['cnn_rp', 'cnn_spp']:
    for atk in ['ndeg']:
        for isd in [0]:
            for label in ['yc']:
                train_params["cnn_model"] = cnn_model
                train_params["atk"] = atk
                train_params["isd"] = isd
                train_params["label"] = label
                model_save_path = f'./checkpoints/{train_params["cnn_model"]}/4_{train_params["atk"]}_isd{train_params["isd"]}_{train_params["label"]}'
                data_load_path = f'./data/train/training_4nets_networks_{train_params["atk"]}_isd{train_params["isd"]}.mat'
                if train_params['label'] in ['pt', 'cc']:
                    output_dim = 1
                else:
                    output_dim = 201

                assert train_params["label"] != 'all'

                if train_params["cnn_model"] == 'cnn_rp':
                    model = CNN_RP(
                        input_size=500,
                        output_size=output_dim,
                    )
                elif train_params["cnn_model"] == 'cnn_spp':
                    model = CNN_SPP(
                        output_size=output_dim,
                    )
                else:
                    model = CNN_LFR(
                        input_size=500,
                        output_size=output_dim,
                    )
                    embedding_load_path = f'./data/lfr_embeddings/train/training_4nets_networks_{train_params["atk"]}_isd{train_params["isd"]}.npy'
                    data_load_path = embedding_load_path
                train_cnn(
                    device=exp_device,
                    model=model,
                    data_path=data_load_path,
                    label=train_params["label"],
                    max_epoch=20,
                    batch_size=1,
                    save_path=model_save_path
                )
                print(f'model: {model_save_path} has saved!')
