import numpy as np
import elder_futhark as ef


def get_data_sets():

    examples = list(ef.runes.values())

    def training_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    def evaluation_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    return training_set, evaluation_set
