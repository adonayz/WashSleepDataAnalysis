from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, LSTM, Dense, Masking
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import prepare_data
from tensorflow.keras.optimizers import Adamax


def generate_training_data():
    return prepare_data.get_actigraphy_model_training_data()


def separate_train_test_set(X, y):
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
    print(trainX.shape)
    # trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    # trainY = trainY.reshape(trainY.shape[0], )
    # testX = testX.reshape(1, testX.shape[0], testX.shape[1])


def save_train_data(X, y):
    with open('frozen_variables/acti_train.npy', 'wb') as f:
        np.save(f, X)
        np.save(f, y)


def load_train_data():
    with open('frozen_variables/acti_train.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)
    return X, y


def save_model(model):
    model.save('ai_models/model_actigraphy_lstm.h5')


def load_model():
    model = tf.saved_model.load('model_actigraphy_lstm.h5')
    return model


def get_data(generate_data_flag):
    if generate_data_flag:
        X, y = generate_training_data()
        save_train_data(X, y)
    else:
        X, y = load_train_data()

    print(X.shape)
    print(y.shape)

    n_steps = X.shape[1]
    n_features = X.shape[2]
    if y.ndim > 1:
        n_output = y.shape[1]
    else:
        n_output = 1

    return X, y, n_steps, n_features, n_output


def make_prediction(model, x_input, n_steps, n_features):
    # demonstrate prediction
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)


def create_model(masking_value=-1, n_steps=100, n_features=131, n_output=1, optimizer='adam', lr=0.001,
                 activation='relu', neurons=50, dropout_rate=0.0, weight_constraint=0, init_mode='uniform'):
    model = Sequential()
    model.add(Masking(mask_value=masking_value, input_shape=(n_steps, n_features)))
    # model.add(LSTM(50, activation=activation, return_sequences=True, kernel_initializer=init_mode))
    model.add(LSTM(neurons, activation=activation, kernel_initializer=init_mode))
    model.add(Dense(n_output))
    optimizer = Adamax(lr=0.001)
    model.compile(loss='mae', optimizer=optimizer, metrics=["accuracy"])

    return model


def fit_model(model, X, y, epochs, batch_size, verbose):
    print(model.summary())
    print("Fitting model...")
    history = model.fit(X, y, epochs=epochs, verbose=verbose)
    print("model fitted")
    print(history.history)


def hyper_parameter_grid_search(X, y, masking_value, n_steps, n_features, n_output):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    model = KerasClassifier(build_fn=create_model, masking_value=masking_value, n_steps=n_steps, n_features=n_features,
                            n_output=n_output, verbose=0)
    # define the grid search parameters
    batch_size = [1, 16, 32, 64, 128, 256, 512]
    epochs = [500, 1000, 1500, 2000]
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                 'he_uniform']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    neurons = [1, 5, 10, 15, 20, 25, 50, 100, 150, 300]
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_grid = dict(batch_size=batch_size, epochs=epochs, activation=activation, neurons=neurons, init_mode=init_mode)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def start_grid_search(mask, g_flag):
    X, y, n_steps, n_features, n_output = get_data(g_flag)

    hyper_parameter_grid_search(X, y, mask, n_steps, n_features, n_output)


def start_training(mask, g_flag):
    verbose, epochs, batch_size = 2, 1500, 32
    opt, lr, actvtn, neurons, dropout_rate, weight_constraint, init_mode = 'adam', 0.001, 'relu', 50, 0.0, 0, 'uniform'

    X, y, n_steps, n_features, n_output = get_data(g_flag)

    model = create_model(masking_value=mask, n_steps=n_steps, n_features=n_features, n_output=n_output,
                         optimizer=opt, lr=lr, activation=actvtn, neurons=neurons, dropout_rate=dropout_rate,
                         weight_constraint=weight_constraint, init_mode=init_mode)
    fit_model(model, X, y, epochs, batch_size, verbose)

    # save_model(model)
    # model = load_model()

    return model, X[0], n_steps, n_features


def start_program():
    masking_value = -1
    generate_data_flag = 1

    start_grid_search(masking_value, generate_data_flag)

    # model, x_input, n_steps, n_features = start_training(masking_value, generate_data_flag)
    # make_prediction(model, x_input, n_steps, n_features)


if __name__ == '__main__':
    start_program()
