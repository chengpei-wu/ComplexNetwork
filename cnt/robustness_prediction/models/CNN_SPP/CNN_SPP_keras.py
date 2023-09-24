import os
import warnings
from time import time

import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model

from SpatialPyramidPooling_tf import SpatialPyramidPooling
from utils.utils_tool_function import Save_loss

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # Ignore warning


class CNN_SPP:
    def __init__(self, epochs=30, ouput_size=201, batch_size=1, valid_proportion=0.1, model=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.ouput_size = ouput_size
        self.metrics = ['mae', 'mse']
        self.optimizer = optimizers.SGD(learning_rate=1e-1)
        self.valid_proportion = valid_proportion
        if not model:
            # initial model for training
            self.model = self.init_model()
        else:
            # initial model for testing
            spp_layer = {'SpatialPyramidPooling': SpatialPyramidPooling}
            self.model = load_model(model, custom_objects=spp_layer)

    def init_model(self):
        model = Sequential()
        # note that the input_shape=(None, None, 1)
        # to allow variable input network sizes
        model.add(Conv2D(64, (7, 7), activation='relu',
                         input_shape=(None, None, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(SpatialPyramidPooling([1, 2, 4]))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(self.ouput_size, activation='hard_sigmoid'))
        model.compile(loss='mean_squared_error',
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        model.summary()
        return model

    def data_generator(self, x, y):
        batches = (len(x) + self.batch_size - 1) // self.batch_size
        while True:
            for i in range(batches):
                tt = x[i * self.batch_size:(i + 1) * self.batch_size]
                Y = y[i * self.batch_size:(i + 1) * self.batch_size]
                X = []
                for t in tt:
                    # adj matrix
                    adj = t.todense()
                    X.append(adj)
                X = np.array(X).reshape(self.batch_size, len(adj), len(adj), 1)
                yield X, Y

    def fit(self, x, y, model_path, save_model=True):
        loss_history = Save_loss()
        on_Plateau = ReduceLROnPlateau(monitor='val_mae', patience=4, factor=0.5, min_delta=1e-3,
                                       verbose=1)
        if save_model:
            filepath = f'{model_path}.hdf5'
            CheckPoint = ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True,
                                         mode='min')
            callbacks_list = [loss_history, CheckPoint]
        else:
            callbacks_list = [loss_history]
        x, y = shuffle(x, y)
        train_size = int(len(x) * (1 - self.valid_proportion))
        valid_size = len(x) - train_size
        x_train, y_train = x[:train_size], y[:train_size]
        x_valid, y_valid = x[train_size:], y[train_size:]
        self.model.fit_generator(
            generator=self.data_generator(x_train, y_train),
            steps_per_epoch=(train_size + self.batch_size -
                             1) // self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks_list,
            verbose=1,
            validation_data=self.data_generator(x_valid, y_valid),
            validation_steps=(valid_size + self.batch_size -
                              1) // self.batch_size
        )
        return np.array(loss_history.losses)

    def my_predict(self, x_test, y_test, y_pt, y_cc, label):
        y_pred = []
        l = len(x_test)
        tic = time()
        for i, x in enumerate(x_test):
            # print_progress(i, l, prefix='CNN-SPP Predicting:')
            adj = x.todense()
            X = np.array(adj).reshape(1, len(adj), len(adj), 1)
            y_pred.append(self.model.predict(np.array(X)))
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


spp = CNN_SPP()
