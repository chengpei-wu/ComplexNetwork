import os
from time import time

import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model

from utils.utils_tool_function import Save_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # Ignore warning


class CNN_RP:
    def __init__(self, input_size, ouput_size=201, epochs=10, batch_size=4, valid_proportion=0.1, model=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_size = input_size
        self.ouput_size = ouput_size
        self.metrics = ['mae', 'mse']
        self.optimizer = optimizers.Adam(
            learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.valid_proportion = valid_proportion
        if not model:
            # initial instance for training
            self.model = self.init_model()
        else:
            # initial instance for testing
            self.model = load_model(model)

    def init_model(self):
        model = Sequential()
        model.add(Conv2D(64, (7, 7), activation='relu', kernel_initializer=initializers.truncated_normal(
            stddev=0.01), input_shape=(self.input_size, self.input_size, 1), padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(64, (5, 5), kernel_initializer=initializers.truncated_normal(
            stddev=0.01), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(128, (3, 3), kernel_initializer=initializers.truncated_normal(
            stddev=0.01), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(128, (3, 3), kernel_initializer=initializers.truncated_normal(
            stddev=0.01), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(256, (3, 3), kernel_initializer=initializers.truncated_normal(
            stddev=0.01), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(256, (3, 3), kernel_initializer=initializers.truncated_normal(
            stddev=0.01), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(512, (3, 3), kernel_initializer=initializers.truncated_normal(
            stddev=0.01), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), kernel_initializer=initializers.truncated_normal(
            stddev=0.01), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(self.ouput_size, activation='relu'))
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
                    adj = t.todense()
                    X.append(adj)
                X = np.array(X).reshape(len(tt),
                                        self.input_size, self.input_size, 1)
                yield X, Y

    def fit(self, x, y, model_path, save_model=True):
        loss_history = Save_loss()
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
            # print_progress(i, l, prefix='CNN-RP Predicting:')
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
