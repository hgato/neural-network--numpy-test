import numpy as np


class Sigmoid:
    def run(self, X):
        sig = 1 / (1 + np.exp(-X))
        sig[sig > 0.9999] = 0.9999
        sig[sig < 0.0001] = 0.0001
        return sig

    def derive(self, A):
        return sigmoid(A) * (1 - sigmoid(A))


class ReLU:
    def run(self, X):
        return np.maximum(0, X)

    def derive(self, A):
        return (A > 0) * 1


class LogLoss:
    def run(self, Y, A):
        m = Y.shape[1]

        # replace zeros with very small value
        one_minus_A = 1 - A
        one_minus_A[one_minus_A == 0] = 0.00000000000001

        log_part = np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(one_minus_A).T)
        return - 1 / m * np.sum(log_part)

    def derive(self, Y, A):
        # replace zeros with very small value
        one_minus_A = 1 - A
        one_minus_A[one_minus_A == 0] = 0.00000000000001

        return - (np.divide(Y, A) - np.divide(1 - Y, one_minus_A))

def sigmoid(X):
    sig = 1 / (1 + np.exp(-X))
    sig[sig > 0.9999] = 0.9999
    sig[sig < 0.0001] = 0.0001
    return sig


def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))


def log_loss(Y, A):
    m = Y.shape[1]

    # replace zeros with very small value
    one_minus_A = 1 - A
    one_minus_A[one_minus_A == 0] = 0.00000000000001

    log_part = np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(one_minus_A).T)
    return - 1/m * np.sum(log_part)


def log_loss_derivative(Y, A):
    # replace zeros with very small value
    one_minus_A = 1 - A
    one_minus_A[one_minus_A == 0] = 0.00000000000001

    return - (np.divide(Y, A) - np.divide(1 - Y, one_minus_A))


def relu(X):
    return np.maximum(0, X)


def relu_derivative(X):
    return (X > 0) * 1