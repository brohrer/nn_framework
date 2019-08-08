import numpy as np

# All of these need to be able to handle 2D numpy arrays as inputs.


class tanh(object):
    @staticmethod
    def calc(v):
        return np.tanh(v)
