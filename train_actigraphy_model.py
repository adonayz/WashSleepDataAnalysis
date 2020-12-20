from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers.core import Activation, Dropout, Dense
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Flatten, LSTM, Dense, Masking

import numpy as np
import pandas as pd
import prepare_data


def generate_training_data():
    return prepare_data.get_actigraphy_model_training_data()


def train_data(X, y):
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
    trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    trainY = trainY.reshape(trainY.shape[0], )
    testX = testX.reshape(1, testX.shape[0], testX.shape[1])


def save_train_data(X, y):
    with open('frozen_variables/acti_train.npy', 'wb') as f:
        np.save(f, X)
        np.save(f, y)


def load_train_data():
    with open('frozen_variables/acti_train.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)
    return X, y


if __name__ == '__main__':
    masking_value = -1
    generate_data_flag = 0

    if generate_data_flag:
        X, y = generate_training_data()
        save_train_data(X, y)
    else:
        X, y = load_train_data()

    print(X.shape)
    print(y.shape)

    n_steps = X.shape[1]
    n_features = X.shape[2]
    # n_output = y.shape[1]
    n_output = 1
    verbose, epochs, batch_size = 2, 200, 32

    model = Sequential()
    model.add(Masking(mask_value=masking_value, input_shape=(n_steps, n_features)))
    model.add(LSTM(1000, activation='relu', return_sequences=True))
    model.add(LSTM(1000, activation='relu'))
    model.add(Dense(n_output))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])

    print(model.summary())

    print("Fitting model...")
    # fit model
    history = model.fit(X, y, validation_split=0.2, epochs=epochs, verbose=verbose)
    print("model fitted")

    print(history.history)

    # model.save('model_lstm.h5')
    # model = keras.load_model('model_lstm.h5')

    # demonstrate prediction
    x_input = X[0]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)
