import random
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Stimulus:
    # stimulus represents a perceptual situation where two exact quantities are given

    def __init__(self, a=None, b=None):
        if a is None:
            self.a = random.randint(0, 101)  # ?
            self.b = random.randint(1, 101)  # ?
        else:
            self.a = a
            self.b = b


class ReactiveUnit:

    sigma = 0.1
    sample_size = 10000

    def __init__(self, stimulus: Stimulus):
        a_samples = self.sample(stimulus.a)
        b_samples = self.sample(stimulus.b)
        self.a = stimulus.a
        self.b = stimulus.b
        # print(stimulus.a/stimulus.b)
        r = a_samples/b_samples
        self.ratio_samples = r.tolist()

    def sample(self, quantity):
        return np.array(np.random.normal(quantity, self.sigma*quantity, self.sample_size), dtype=np.float)

    def show(self, how="spline"):
        plt.title("Reactive unit for " + str(self.a) + "/" + str(self.b) + " = " + str(self.a / self.b))
        if how == "hist":
            plt.hist(self.ratio_samples, bins="auto")
            plt.show()
        elif how == "spline":
            y, bin_edges = np.histogram(self.ratio_samples, bins=45)
            # x = np.arange((bin_edges[0] + bin_edges[1])/2, bin_edges[-1], bin_edges[1] - bin_edges[0])
            x = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(0, len(bin_edges)-1)]
            f = interp1d(x, y)
            f2 = interp1d(x, y, kind="cubic")
            plt.plot(x, y, 'o', x, f(x), '-', x, f2(x), '--')
            plt.legend(['data', 'linear', 'cubic'], loc='best')
            plt.show()

class DiscriminativeCategory:

    def __init__(self, weights=None, reactive_units=None):
        if weights is None:
            self.weights = []
            self.reactive_units = []
        else:
            self.weights = weights
            self.reactive_units = reactive_units

    def response(self, stimulus: Stimulus):
        #ru = ReactiveUnit(stimulus)
        #total_ratios = ru.ratios + self.get_all_ratios()
        #total_hist, total_bin_edges = np.histogram(total_ratios, bins="auto")
        #bin_width = total_bin_edges[1] - total_bin_edges[0]
        #print(stimulus.a, stimulus.b)
        #plt.hist(ru.ratios, bins=[i for i in np.arange(1,3,0.007)])
        #plt.title("Representation of stimulus")
        #plt.show()
        pass

    def add_reactive_unit(self, reactive_unit: ReactiveUnit, weight=0.5):
        self.weights.append(weight)
        self.reactive_units.append(reactive_unit)

    def select(self, stimuli):
        return 0

    def update_weights(self):
        pass

    def show(self):
        flat = self.get_flat()
        plt.hist(flat, bins="auto")
        plt.title("category")
        plt.show()

    def get_flat(self):
        flat_ratios = []
        flat_weights = []
        for i in range(0, len(self.reactive_units)):
            flat_ratios += self.reactive_units[i].ratios
            flat_weights += [self.weights[i]]*len(self.reactive_units[i].ratios)
        return flat_ratios, flat_weights