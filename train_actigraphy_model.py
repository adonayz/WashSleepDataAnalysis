from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers.core import Activation, Dropout, Dense
from keras.models import Sequential
from keras.layers import Flatten, LSTM

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


if __name__ == '__main__':
    X, y = generate_training_data()
    n_steps = X.shape[1]
    n_features = X.shape[2]
    n_output = y.shape[1]
    verbose, epochs, batch_size = 0, 50, 64

    print("Window: " + str(n_steps))
    print(X.shape)
    print(y.shape)
    print(n_features)

    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')

    print(model.summary())

    print("Fitting model...")
    # fit model
    history = model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=verbose)
    print("model fitted")

    print(history.history)

    # model.save('model_lstm.h5')
    # model = keras.load_model('model_lstm.h5')

    # demonstrate prediction
    x_input = X[0]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)
