import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")


class ANN(object):
    def __init__(
        self,
        model=None,
        expected_range=(-1, 1),
    ):
        self.layers = model
        self.n_iter_train = int(1e8)
        self.n_iter_evaluate = int(1e6)
        self.expected_range = expected_range

    def train(self, training_set):
        for i_iter in range(self.n_iter_train):
            x = self.normalize(next(training_set()).ravel())
            y = self.forward_prop(x)
            print(y)

    def evaluate(self, evaluation_set):
        for i_iter in range(self.n_iter_evaluate):
            x = self.normalize(next(evaluation_set()).ravel())
            y = self.forward_prop(x)

    def forward_prop(self, x):
        # Convert the inputs into a 2D array of the right shape.
        y = x.ravel()[np.newaxis, :]
        self.layers[0].forward_prop(y)
        return y.ravel()

    def normalize(self, values):
        """
        Transform the input/output values so that they tend to
        fall between -.5 and .5
        """
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = max_val - min_val
        offset_factor = min_val
        return (values - offset_factor) / scale_factor - .5

    def denormalize(self, transformed_values):
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = 2 / (max_val - min_val)
        offset_factor = min_val - 1
        return transformed_values / scale_factor - offset_factor
