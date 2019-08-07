import data_loader_two_by_two as dat
import nn_framework.framework as framework

training_set, evaluation_set = dat.get_data_sets()

sample = next(training_set())
input_value_range = (0, 1)
n_pixels = sample.shape[0] * sample.shape[1]

n_nodes = [n_pixels, n_pixels]

autoencoder = framework.ANN(
    model=None,
    expected_range=input_value_range,
)
autoencoder.train(training_set)
autoencoder.evaluate(evaluation_set)
