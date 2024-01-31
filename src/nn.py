import numpy as np
from src.functions import Sigmoid, ReLU, LogLoss
from src.utilities.utility_classes import SavableModel


class NeuralNetwork(SavableModel):
    # TODO separate classes
    def __init__(self, options):
        self.options = options
        self.parameters = {
            'W': {},
            'b': {},
        }
        self.cache = {
            'A': {},
            'Z': {},
        }
        self.activation_functions = {}
        # TODO use layers with options instead of label layers
        self.layer_labels = [str(i + 1) for i in range(0, len(self.options['layers']))]
        self.derivatives = {
            'dZ': {},
            'dA': {},
            'dW': {},
            'db': {},
        }

        self._initialize_weights_biases()
        self._set_activation_functions()

    def _initialize_weights_biases(self):
        for option_layer_index in range(0, len(self.options['layers'])):
            layer_number = self.layer_labels[option_layer_index]

            n2 = self.options['layers'][option_layer_index]['n']
            if option_layer_index == 0:
                n1 = self.options['input_shape'][0]
            else:
                n1 = self.options['layers'][option_layer_index - 1]['n']

            w_shape = (n2, n1)
            b_shape = (n2, 1)

            default_weight_multiplier = self.options['layers'][option_layer_index]['default_weight_multiplier']
            # TODO add options for weight initialization
            self.parameters['W'][layer_number] = np.random.randn(w_shape[0], w_shape[1]) * default_weight_multiplier
            self.parameters['b'][layer_number] = np.zeros(b_shape)

    def _set_activation_functions(self):
        for option_layer_index in range(0, len(self.options['layers'])):
            layer_number = self.layer_labels[option_layer_index]
            activation_function_label = self.options['layers'][option_layer_index]['activation_function']

            if activation_function_label == 'sigmoid':
                function_object = Sigmoid()
            elif activation_function_label == 'relu':
                function_object = ReLU()
            else:
                raise Exception('Unknown activation function: ' + activation_function_label)

            self.activation_functions[layer_number] = function_object

    def fit(self, X, Y, num_iterations):
        # TODO add batch functionality
        # TODO decide if num_iterations should be set in options
        for iteration_number in range(num_iterations + 1):
            A = X
            self.cache['A']['0'] = X
            for l in self.layer_labels:
                A = self._forward_for_layer(A, l)
            loss, dA = self._count_loss(Y, A)
            self._print_loss(loss, iteration_number, num_iterations)
            for l in self.layer_labels[::-1]:
                # TODO remove shape attribute
                dA = self._backward_for_layer(dA, l, X.shape[1])

    def predict(self, X):
        A = X
        for l in self.layer_labels:
            A = self._forward_for_layer(A, l)
        return A

    def _print_loss(self, loss, iteration_number, num_iterations):
        if iteration_number % 100 == 0 or iteration_number == num_iterations:
            print('Loss after ', iteration_number, ' iterations: ', loss)

    def _forward_for_layer(self, A, l):
        activation_function = self._get_activation_function(l)
        W = self.parameters['W'][l]
        b = self.parameters['b'][l]

        A, Z = self._calculate_forward(A, W, b, activation_function)

        self.cache['Z'][l] = Z
        self.cache['A'][l] = A

        return A

    def _calculate_forward(self, A, W, b, activation_function):
        Z = np.dot(W, A) + b
        A = activation_function.run(Z)
        return A, Z

    def _backward_for_layer(self, dA, l, m):
        activation_function = self._get_activation_function(l)
        Z = self.cache['Z'][l]
        W = self.parameters['W'][l]
        A_prev = self.cache['A'][str(int(l)-1)]

        dZ, dW, db, dA_prev = self._calculate_backward(dA, Z, m, A_prev, W, activation_function)

        self._update_W_b_for_layer(dW, db, l)
        return dA_prev

    def _calculate_backward(self, dA, Z, m, A_prev, W, activation_function):
        dZ = dA * activation_function.derive(Z)
        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = np.dot(W.T, dZ)
        return dZ, dW, db, dA_prev

    def _update_W_b_for_layer(self, dW, db, l):
        # TODO remove hardcode
        learning_rate = 0.0075
        W = self.parameters['W'][l]
        b = self.parameters['b'][l]
        W = W - learning_rate * dW
        b = b - learning_rate * db
        self.parameters['W'][l] = W
        self.parameters['b'][l] = b

    def _count_loss(self, Y, A):
        loss_function = self._get_loss_function()
        loss = loss_function.run(Y, A)
        dA = loss_function.derive(Y, A)
        return loss, dA

    def _get_activation_function(self, l):
        return self.activation_functions[l]

    def _get_loss_function(self):
        if self.options['loss_function'] == 'log_loss':
            return LogLoss()
        else:
            raise Exception('Unknown loss function: ' + self.options['loss_function'])

