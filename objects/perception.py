import random
import numpy as np
# import matplotlib.pyplot as plt


class Stimulus:
    # stimulus represents a perceptual situation where two exact quantities are given

    def __init__(self):
        self.a = random.randint(0, 101)  # ?
        self.b = random.randint(1, 101)  # ?


class ReactiveUnit:

    sigma = 0.1
    sample_size = 1000

    def __init__(self, stimulus: Stimulus):
        representation1 = self.get_representation(stimulus.a)
        representation2 = self.get_representation(stimulus.b)
        # print(stimulus.a/stimulus.b)
        self.ratios = representation1/representation2

    def get_representation(self, quantity):
        return np.array(np.random.normal(quantity, self.sigma, self.sample_size), dtype=np.float)


class DiscriminativeCategory:

    def __init__(self, weights=None, reactive_units=None):
        if weights is None:
            self.weights = []
            self.reactive_units = []
        else:
            self.weights = weights
            self.reactive_units = reactive_units

    def response(self, stimulus: Stimulus):
        ru = ReactiveUnit(stimulus)
        hist, bin_edges = np.histogram(ru.ratios, bins="auto")
        bin_width = bin_edges[1] - bin_edges[0]
        # plt.hist(ru.ratios, bins="auto")
        # plt.title("Representation of stimulus")
        # plt.show()

    def add_reactive_unit(self, reactive_unit: ReactiveUnit, weight=0.5):
        self.weights.append(weight)
        self.reactive_units.append(reactive_unit)

    def select(self, stimuli):
        return 0

    def update_weights(self):
        pass