import time

from models.CNN_LFR.get_tensors import get_lfr_tensors
from parameters import *

train_params["atk"] = 'ndeg'
train_params["isd"] = 0
data_load_path = f'./data/train/training_4nets_networks_{train_params["atk"]}_isd{train_params["isd"]}.mat'
embedding_save_path = f'./data/lfr_embeddings/train/training_4nets_networks_{train_params["atk"]}_isd{train_params["isd"]}'
print(embedding_save_path)
tic = time.time()
get_lfr_tensors(data_path=data_load_path, save_path=embedding_save_path)
toc = (time.time() - tic) / 900
print(f'{range}: {toc}')

data_load_path = f'./data/test/testing_4nets_networks_{train_params["atk"]}_isd{train_params["isd"]}.mat'
embedding_save_path = f'./data/lfr_embeddings/test/testing_4nets_networks_{train_params["atk"]}_isd{train_params["isd"]}'
print(embedding_save_path)
tic = time.time()
get_lfr_tensors(data_path=data_load_path, save_path=embedding_save_path)
toc = (time.time() - tic) / 900
print(f'{range}: {toc}')
