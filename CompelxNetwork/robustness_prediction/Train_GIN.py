from models.GIN import Multi_GIN, GIN
from utils.utils_gnn import calculate_param_number
from utils.utils_train_test import *

if __name__ == '__main__':
    exp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for atk in ['ndeg']:
        for isd in [0]:
            for label in ['all', 'yc']:
                train_params["atk"] = atk
                train_params["isd"] = isd
                train_params["label"] = label
                if train_params['label'] in ['pt', 'cc']:
                    output_dim = 1
                else:
                    output_dim = 201
                # train model
                model_save_path = f'./checkpoints/gin/real_{train_params["atk"]}_isd{train_params["isd"]}_{train_params["label"]}'
                data_load_path = f'./data/train/training_real_networks_{train_params["atk"]}_isd{train_params["isd"]}.mat'
                if train_params['label'] == 'all':
                    mul_gin = Multi_GIN(
                        input_dim=1,
                        hidden_dim=16,
                        output_dim=output_dim,
                    )
                    calculate_param_number(mul_gin)
                    loss_history = train_multi_gnn(
                        device=exp_device,
                        model=mul_gin,
                        data_path=data_load_path,
                        label=train_params["label"],
                        max_epoch=train_params["max_epoch"],
                        batch_size=train_params["batch_size"],
                        save_path=model_save_path
                    )
                    np.save('./mul_gin_losses.npy', loss_history)
                else:
                    gin = GIN(
                        input_dim=1,
                        hidden_dim=16,
                        output_dim=output_dim,
                    )
                    calculate_param_number(gin)
                    loss_history = train_gnn(
                        device=exp_device,
                        model=gin,
                        data_path=data_load_path,
                        label=train_params["label"],
                        max_epoch=train_params["max_epoch"],
                        batch_size=train_params["batch_size"],
                        save_path=model_save_path
                    )
                    np.save('./gin_losses.npy', loss_history)
