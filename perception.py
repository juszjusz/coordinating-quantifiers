from random import randint
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Stimulus:
    # stimulus represents a perceptual situation where two exact quantities are given

    def __init__(self, a=randint(0, 101), b=randint(1, 101)):
        self.a = a
        self.b = b


class ReactiveUnit:
    sigma = 0.1
    sample_size = 10000

    def __init__(self, stimulus):
        # TODO consider different interpolation methods
        a_samples = self.sample(stimulus.a)
        b_samples = self.sample(stimulus.b)
        self.a = stimulus.a
        self.b = stimulus.b
        r = a_samples / b_samples
        self.ratio_samples = r.tolist()
        y, bin_edges = np.histogram(self.ratio_samples, bins="auto", density=True)
        x = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(0, len(bin_edges) - 1)]
        self.interp = interp1d(x, y) #radial basis interpolation?
        self.x_left = x[0]
        self.x_right = x[-1]

    def sample(self, quantity):
        return np.array(np.random.normal(quantity, self.sigma * quantity, self.sample_size), dtype=np.float)

    def fun(self, x):
        # TODO refactor
        if type(x) is np.ndarray:
            y = np.empty_like(x)
            for i in range(len(x)):
                y[i] = 0 if x[i] < self.x_left or x[i] > self.x_right else self.interp(x[i])
            return y
        else:
            return 0 if x < self.x_left or x > self.x_right else self.interp(x)

    def show(self, how="spline"):
        plt.title("Reactive unit for " + str(self.a) + "/" + str(self.b) + " = " + str(self.a / self.b))
        if how == "hist":
            plt.hist(self.ratio_samples, bins="auto")
            plt.show()
        elif how == "spline":
            x = np.linspace(self.x_left, self.x_right)
            plt.plot(x, self.fun(x), 'o', x, self.fun(x), '--')
            plt.legend(['data', 'cubic'], loc='best')
            plt.hist(self.ratio_samples, bins=50)
            plt.show()


class Category:

    def __init__(self):
        self.weights = []
        self.reactive_units = []
        self.x_left = float("inf")
        self.x_right = float("-inf")

    def response(self, stimulus):
        s = ReactiveUnit(stimulus)
        x_left = min(s.x_left, self.x_left)
        x_right = max(s.x_right, self.x_right)
        return scipy.integrate.quad(lambda x: s.fun(x)*self.fun(x), x_left, x_right)[0]

    def add_reactive_unit(self, reactive_unit, weight=0.5):
        self.weights.append(weight)
        self.reactive_units.append(reactive_unit)
        self.x_left = min(self.x_left, reactive_unit.x_left)
        self.x_right = max(self.x_right, reactive_unit.x_right)

    def fun(self, x):
        # performance?
        return 0 if len(self.reactive_units) == 0 \
            else sum([r.fun(x)*w for r, w in zip(self.reactive_units, self.weights)])

    def select(self, stimuli):
        responses = [self.response(s) for s in stimuli]
        max_response = max(responses)
        which = [i for i, j in enumerate(responses) if j == max_response]
        return which[0] if len(which) == 1 else None

    def update_weights(self, factors):
        self.weights = [weight + factor*weight for weight, factor in zip(self.weights, factors)]

    def show(self):
        x = np.linspace(self.x_left, self.x_right, num=100)
        plt.plot(x, self.fun(x), 'o', x, self.fun(x), '--')
        plt.legend(['data', 'cubic'], loc='best')
        plt.show()

    def get_flat(self):
        # TODO
        #flat_ratios = []
        #flat_weights = []
        #for i in range(0, len(self.reactive_units)):
        #    flat_ratios += self.reactive_units[i].ratios
        #    flat_weights += [self.weights[i]] * len(self.reactive_units[i].ratios)
        #return flat_ratios, flat_weights
        return []
