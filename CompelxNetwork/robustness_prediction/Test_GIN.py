from torch.utils.data import DataLoader

from parameters import test_params
from utils.utils_gnn import *
from utils.utils_train_test import predict_multi_gnn, predict

if __name__ == '__main__':

    exp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for atk in ['ndeg']:
        for isd in [0]:
            for label in ['all']:
                test_params["atk"] = atk
                test_params["isd"] = isd
                test_params["label"] = label
                load_model_path = f'./checkpoints/gin/real_{test_params["atk"]}_isd{test_params["isd"]}_{test_params["label"]}'
                load_data_path = f'./data/test/testing_real_networks_{test_params["atk"]}_isd{test_params["isd"]}'
                save_pred_path = f'./prediction/gin/real_{test_params["atk"]}_isd{test_params["isd"]}_{test_params["label"]}'
                graphs = load_network_gnn(load_data_path, test_params['label'])
                gin = torch.load(load_model_path, map_location=exp_device)
                if test_params["label"] == 'all':
                    test_loader = DataLoader(
                        graphs,
                        batch_size=512,
                        collate_fn=collate_gnn_multi,
                    )
                    prediction = predict_multi_gnn(test_loader, exp_device, gin)
                else:
                    test_loader = DataLoader(
                        graphs,
                        batch_size=512,
                        collate_fn=collate_gnn,
                    )
                    prediction = predict(test_loader, exp_device, gin)
                sio.savemat(save_pred_path + '.mat', prediction)
                print(f'prediction results saved in: {save_pred_path}')
