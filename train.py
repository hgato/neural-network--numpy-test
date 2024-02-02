import datetime

import h5py
from src.nn import NeuralNetwork


def load_dataset():
    file = 'data/train_data.h5'
    data = h5py.File(file, 'r')

    X = data['train_set_x'][:]
    Y = data['train_set_y'][:]

    X = X.reshape((X.shape[0], -1)).T / 255
    Y = Y.reshape((Y.shape[0], 1)).T
    return X, Y


if __name__ == '__main__':
    X, Y = load_dataset()
    # TODO make options and layers classes
    example_options = {
        'num_features': X.shape[0],
        'layers': [
            {
                'n': 32,
                'initial_weight_multiplier': 'he',
                'activation_function': 'relu',
                'learning_rate': 0.001,
                'keep_probability': 1.0,
            },
            {
                'n': 8,
                'initial_weight_multiplier': 'he',
                'activation_function': 'relu',
                'learning_rate': 0.001,
                'keep_probability': 1.0,
            },
            {
                'n': 1,
                'initial_weight_multiplier': 'he',
                'activation_function': 'sigmoid',
                'learning_rate': 0.001,
                'keep_probability': 1.0,
            },
        ],
        'loss_function': 'log_loss'
    }
    nn = NeuralNetwork(example_options)
    nn.fit(X, Y, 7500)
    nn.save('save/{}.pickle'.format(str(datetime.date.today())))
