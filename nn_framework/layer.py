import numpy as np


class Dense(object):
    def __init__(
        self,
        m_inputs,
        n_outputs,
        debug=False,
    ):
        self.debug = debug
        self.m_inputs = int(m_inputs)
        self.n_outputs = int(n_outputs)
        self.learning_rate = .05

        # Choose random weights.
        # Inputs match to rows. Outputs match to columns.
        # Add one to m_inputs to account for the bias term.
        # self.initial_weight_scale = 1 / self.m_inputs
        self.initial_weight_scale = 1
        self.weights = self.initial_weight_scale * (np.random.sample(
            size=(self.m_inputs + 1, self.n_outputs)) * 2  - 1)
        self.w_grad = np.zeros((self.m_inputs + 1, self.n_outputs))
        self.x = np.zeros((1, self.m_inputs + 1))
        self.y = np.zeros((1, self.n_outputs))
