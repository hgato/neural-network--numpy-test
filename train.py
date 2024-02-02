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
    example_options = {
        'num_features': X.shape[0],
        'layers': [
            {
                'n': 32,
                'default_weight_multiplier': 1,
                'activation_function': 'relu',
                'learning_rate': 0.001,
            },
            {
                'n': 8,
                'default_weight_multiplier': 1,
                'activation_function': 'relu',
                'learning_rate': 0.001,
            },
            {
                'n': 1,
                'default_weight_multiplier': 1,
                'activation_function': 'sigmoid',
                'learning_rate': 0.001,
            },
        ],
        'loss_function': 'log_loss'
    }
    nn = NeuralNetwork(example_options)
    nn.fit(X, Y, 7500)
    nn.save('save/{}.pickle'.format(str(datetime.date.today())))
