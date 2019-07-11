import numpy as np

# All of these expect two identically sized numpy arrays as inputs
# and return the same size error output.


class abs(object):
    @staticmethod
    def calc(x, y):
        return np.abs(y - x)

    @staticmethod
    def calc_d(x, y):
        return -np.sign(y - x)


class sqr(object):
    def calc(self, x, y):
        return np.sign(y - x) * (y - x)**2

    def cacl_d(self, x, y):
        return -np.sign(y - x) * 2 * x
