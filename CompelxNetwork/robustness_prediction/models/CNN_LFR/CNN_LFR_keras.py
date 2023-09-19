from time import time

import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from utils.utils_cnn import Save_loss
from utils.utils_gnn import print_progress


class CNN_LFR:
    def __init__(self, input_size=1000, output_size=201, epochs=50, batch_size=16, model=None):
        self.epochs = epochs
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.metrics = ['mse', 'mae']
        self.optimizer = Adam(lr=0.0001)
        if not model:
            # initial model for training
            self.model = self.init_model()
        else:
            # initial model for testing
            self.model = load_model(model)

    def init_model(self):
        model = Sequential()
        if self.input_size == 500:
            model.add(Conv2D(64, (7, 7), activation='relu',
                             input_shape=(100, 100, 1)))
        if self.input_size == 1000:
            model.add(Conv2D(64, (7, 7), activation='relu',
                             input_shape=(125, 160, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(201, activation='relu'))
        model.compile(loss='mean_squared_error',
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        model.summary()
        return model

    def fit(self, x_embeddings, y, model_path, save_model=True):
        loss_history = Save_loss()
        if save_model:
            filepath = f'{model_path}' + '.hdf5'
            CheckPoint = ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True,
                                         mode='min')
            callbacks_list = [loss_history, CheckPoint]
        else:
            callbacks_list = [loss_history]
        x_embeddings, y = shuffle(x_embeddings, y)
        self.model.fit(x=x_embeddings,
                       y=y,
                       batch_size=self.batch_size,
                       validation_split=0.1,
                       epochs=self.epochs,
                       shuffle=False,
                       callbacks=callbacks_list,
                       verbose=1)
        return np.array(loss_history.losses)

    def my_predict(self, x_test, y_test, y_pt, y_cc, label):
        y_pred = []
        l = len(x_test)
        tic = time()
        for i, x in enumerate(x_test):
            print_progress(i, l, prefix='CNN-LFR Predicting:')
            x = x.reshape(1, x.shape[0], x.shape[1], 1)
            y_pred.append(self.model.predict(x))
        toc = time() - tic
        y_pred = np.array(y_pred).squeeze()
        y_test = np.array(y_test).squeeze()
        prediction = {
            f'pred_{label}': y_pred,
            f'sim_{label}': y_test,
            f'mae_{label}': np.mean(np.abs(y_test - y_pred), axis=1),
            'time': toc / l,
            'sim_t': np.round(np.round(np.array(y_pt).squeeze(), 3) * 200) / 2,
            'sim_m': np.round(np.array(y_cc).squeeze(), 3) * 100,
        }
        return prediction
