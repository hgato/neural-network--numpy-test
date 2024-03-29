import numpy as np
from src.functions import Sigmoid, ReLU, LogLoss
from src.utilities.utility_classes import SavableModel, PrintableModel


class NeuralNetwork(SavableModel, PrintableModel):
    def __init__(self, options):
        self.options = options
        self.layers = []
        self.parameters = {
            'W': {},
            'b': {},
        }
        self.cache = {
            'A': {},
            'Z': {},
            'D': {},
        }
        self.activation_functions = {}
        self.derivatives = {
            'dZ': {},
            'dA': {},
            'dW': {},
            'db': {},
        }

        self._initialize_layers()
        self._initialize_weights_biases()

    def fit(self, X, Y, num_iterations):
        # TODO add batch functionality
        # TODO decide if num_iterations should be set in options
        for iteration_number in range(num_iterations + 1):
            A = X
            self.cache['A']['0'] = X
            for layer in self.layers:
                A = self._forward_for_layer(A, layer)
            loss, dA = self._count_loss(Y, A)
            self._print_loss(loss, iteration_number, num_iterations)
            for layer in self.layers[::-1]:
                dA = self._backward_for_layer(dA, layer)

    def predict(self, X):
        A = X
        for layer in self.layers:
            A = self._forward_for_layer(A, layer)
        return A

    def _forward_for_layer(self, A, layer):
        l = layer['l']
        W = self.parameters['W'][l]
        b = self.parameters['b'][l]

        A, Z = self._calculate_forward(A, W, b, layer['activation_function'])
        A, D = self._apply_dropout_forward(A, layer['keep_probability'])

        self.cache['Z'][l] = Z
        self.cache['A'][l] = A
        self.cache['D'][l] = D

        return A

    def _calculate_forward(self, A, W, b, activation_function):
        Z = np.dot(W, A) + b
        A = activation_function.run(Z)
        return A, Z

    def _apply_dropout_forward(self, A, keep_probability):
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_probability).astype(int)
        A = A * D
        A /= keep_probability
        return A, D

    def _backward_for_layer(self, dA, layer):
        m = dA.shape[1]
        l = layer['l']
        Z = self.cache['Z'][l]
        D = self.cache['D'][l]
        W = self.parameters['W'][l]
        A_prev = self.cache['A'][str(int(l)-1)]

        dA = self._apply_dropout_backward(dA, layer, D)
        dZ, dW, db, dA_prev = self._calculate_backward(dA, Z, m, A_prev, W, layer['activation_function'])

        self._update_W_b_for_layer(dW, db, layer)
        return dA_prev

    def _calculate_backward(self, dA, Z, m, A_prev, W, activation_function):
        dZ = dA * activation_function.derive(Z)
        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        #regularization
        dW += (self.options['regularization_strength'] / m * W)

        dA_prev = np.dot(W.T, dZ)
        return dZ, dW, db, dA_prev

    def _apply_dropout_backward(self, dA_prev, layer, D):
        dA_prev = dA_prev * D
        dA_prev /= layer['keep_probability']
        return dA_prev

    def _update_W_b_for_layer(self, dW, db, layer):
        learning_rate = layer['learning_rate']
        l = layer['l']

        W = self.parameters['W'][l]
        b = self.parameters['b'][l]

        W, b = self._calculate_update_W_b(W, dW, b, db, learning_rate)

        self.parameters['W'][l] = W
        self.parameters['b'][l] = b

    def _calculate_update_W_b(self, W, dW, b, db, learning_rate):
        W = W - learning_rate * dW
        b = b - learning_rate * db
        return W, b

    def _count_loss(self, Y, A):
        loss_function = self._get_loss_function()
        loss = loss_function.run(Y, A)

        # regularization
        loss += (1 / Y.shape[1] *
                 self.options['regularization_strength'] / 2 *
                 sum([np.sum(W) for W in self.parameters['W'].values()]))

        dA = loss_function.derive(Y, A)
        return loss, dA

    def _get_activation_function(self, l):
        return self.activation_functions[l]

    def _get_loss_function(self):
        if self.options['loss_function'] == 'log_loss':
            return LogLoss()
        else:
            raise Exception('Unknown loss function: ' + self.options['loss_function'])

    def _initialize_layers(self):
        for i in range(len(self.options['layers'])):
            layer = self.options['layers'][i]

            layer['l'] = str(i + 1)
            layer['activation_function'] = self._init_activation_function_object(layer['activation_function'])
            self.layers.append(layer)

    def _init_activation_function_object(self, name):
        if name == 'sigmoid':
            return Sigmoid()
        elif name == 'relu':
            return ReLU()
        else:
            raise Exception('Unknown activation function: ' + name)

    def _initialize_weights_biases(self):
        prev_layer = None
        for layer in self.layers:
            n2 = layer['n']
            if not prev_layer:
                n1 = self.options['num_features']
            else:
                n1 = prev_layer['n']

            w_shape = (n2, n1)
            b_shape = (n2, 1)

            self.parameters['W'][layer['l']] = np.random.randn(w_shape[0], w_shape[1]) * self._get_initial_weight_multiplier(layer, w_shape)
            self.parameters['b'][layer['l']] = np.zeros(b_shape)
            prev_layer = layer

    def _get_initial_weight_multiplier(self, layer, w_shape):
        init_mult = layer['initial_weight_multiplier']
        if type(init_mult) is int or type(init_mult) is float:
            return init_mult
        if init_mult == 'he':
            return np.sqrt(2 / w_shape[1])
        raise Exception('Unknown weight multiplier: ' + str(init_mult))
