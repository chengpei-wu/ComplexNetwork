import tensorflow as tf

from parameters import train_params
from utils_cnn import load_network_with_fixed_size, load_network, load_labels, load_lfr_embeddings
from .CNN_LFR import CNN_LFR
from .CNN_RP import CNN_RP
from .CNN_SPP import CNN_SPP

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model_save_path = f'./checkpoints/...'
data_load_path = f'./data/train/...'
output_dim = 21
if train_params["cnn_model"] == 'cnn_rp':
    model = CNN_RP(
        input_size=1000,
        ouput_size=output_dim,
        epochs=10,
        batch_size=2,
        valid_proportion=0.1,
        model=None
    )
    x, y = load_network_with_fixed_size(data_load_path, train_params["label"], 500)
elif train_params["cnn_model"] == 'cnn_spp':
    model = CNN_SPP(
        epochs=10,
        batch_size=1,
        ouput_size=output_dim,
        valid_proportion=0.1,
        model=None
    )
    x, y = load_network(data_load_path, train_params["label"])
elif train_params["cnn_model"] == 'cnn_lfr':
    model = CNN_LFR(
        input_size=1000,
        output_size=output_dim,
        epochs=200,
        batch_size=16
    )
    embedding_load_path = f'./data/lfr_embeddings/train/...'
    x = load_lfr_embeddings(embedding_load_path, w=500)
    y = load_labels(data_load_path, train_params["label"])
    print(x.shape, y.shape)
else:
    raise NotImplementedError

model.fit(x, y, model_save_path)
print(f'model: {model_save_path} has saved!')
