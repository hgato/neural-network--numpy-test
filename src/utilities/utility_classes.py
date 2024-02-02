import pickle


class SavableModel:
    def __init__(self, options):
        self.parameters = {}
        self.options = options

    def save(self, file):
        pickle.dump({'parameters': self.parameters, 'options': self.options}, open(file, 'wb'))

    @classmethod
    def load(cls, file):
        data = pickle.load(open(file, 'rb'))
        nn = cls(data['options'])
        nn.parameters = data['parameters']
        return nn


class PrintableModel:
    def _print_loss(self, loss, iteration_number, num_iterations):
        if iteration_number % 100 == 0 or iteration_number == num_iterations:
            print('Loss after ', iteration_number, ' iterations: ', loss)