import datetime
import h5py
import numpy as np

from src.nn import NeuralNetwork


def load_dataset():
    file = 'data/test_data.h5'
    data = h5py.File(file, 'r')

    X = data['test_set_x'][:]
    Y = data['test_set_y'][:]

    X = X.reshape((X.shape[0], -1)).T / 255
    Y = Y.reshape((Y.shape[0], 1)).T
    return X, Y


if __name__ == '__main__':
    X, Y = load_dataset()
    nn = NeuralNetwork.load('save/{}.pickle'.format(str(datetime.date.today())))
    A = nn.predict(X)
    A = A > 0.5
    A = A.astype(int)

    compare = A == Y
    print(np.sum(compare) / Y.shape[1] * 100)