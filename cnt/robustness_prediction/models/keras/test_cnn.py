import scipy.io as sio

from parameters import test_params
from utils_cnn import load_network_with_fixed_size, load_network, load_labels, load_lfr_embeddings
from .CNN_LFR import CNN_LFR
from .CNN_RP import CNN_RP
from .CNN_SPP import CNN_SPP

prefix = ''

for size in [200]:
    for cnn_model in [
        'cnn_lfr',
        'cnn_spp',
        'cnn_rp'
    ]:
        for atk in ['ndeg', 'nrnd']:
            for isd in [0, 1]:
                for label in ['lc', 'yc']:
                    test_params["cnn_model"] = cnn_model
                    test_params["atk"] = atk
                    test_params["isd"] = isd
                    test_params["label"] = label
                    load_model_path = f'./checkpoints/{test_params["cnn_model"]}/{prefix}{test_params["atk"]}_isd{test_params["isd"]}_{test_params["label"]}.hdf5'
                    load_data_path = f'./data/test/testing_{prefix}networks_{test_params["atk"]}_isd{test_params["isd"]}_{size}.mat'
                    save_pred_path = f'./prediction/{test_params["cnn_model"]}/{prefix}{test_params["atk"]}_isd{test_params["isd"]}_{test_params["label"]}_{size}.mat'
                    if test_params['label'] in ['pt', 'cc']:
                        output_dim = 1
                    else:
                        output_dim = 20

                    assert test_params["label"] != 'all'

                    if test_params["cnn_model"] == 'cnn_rp':
                        model = CNN_RP(
                            input_size=1000,
                            # ouput_size=output_dim,
                            # epochs=20,
                            # batch_size=4,
                            # valid_proportion=0.1,
                            model=load_model_path
                        )
                        x, y = load_network_with_fixed_size(load_data_path, test_params["label"], 1000)
                    elif test_params["cnn_model"] == 'cnn_spp':
                        model = CNN_SPP(
                            # epochs=20,
                            # batch_size=1,
                            # ouput_size=output_dim,
                            # valid_proportion=0.1,
                            model=load_model_path
                        )
                        x, y = load_network(load_data_path, test_params["label"])
                    else:
                        model = CNN_LFR(
                            input_size=1000,
                            # output_size=output_dim,
                            # epochs=200,
                            # batch_size=test_params["batch_size"],
                            model=load_model_path
                        )
                        embedding_load_path = f'./data/lfr_embeddings/test/testing_{prefix}networks_{test_params["atk"]}_isd{test_params["isd"]}_{size}.npy'
                        print(embedding_load_path)
                        x = load_lfr_embeddings(embedding_load_path, w=1000)
                        y = load_labels(load_data_path, test_params["label"])
                    y_pt = load_labels(load_data_path, 'pt')
                    y_cc = load_labels(load_data_path, 'cc')
                    print(x.shape)
                    prediction = model.my_predict(x_test=x, y_test=y, y_pt=y_pt, y_cc=y_cc, label=test_params['label'])
                    sio.savemat(save_pred_path, prediction)
                    print(f'\nprediction results saved in: {save_pred_path}')
