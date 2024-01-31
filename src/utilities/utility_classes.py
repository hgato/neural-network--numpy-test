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