import numpy as np


class ANN(object):
    def __init__(
        self,
        model=None,
    ):
        self.layers = model
        self.n_iter_train = int(1e8)
        self.n_iter_evaluate = int(1e6)

    def train(self, training_set):
        for i_iter in range(self.n_iter_train):
            x = next(training_set()).ravel()
            print(x)

    def evaluate(self, evaluation_set):
        for i_iter in range(self.n_iter_evaluate):
            x = next(evaluation_set()).ravel()
