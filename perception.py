from __future__ import division # force python 3 division in python 2
from random import randint
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from visualization import Viewable
from numpy import linspace
from matplotlib.ticker import ScalarFormatter
import logging


class Error:
    NO_ERROR = -1
    ERROR = -2
    _END_ = ERROR


class Stimulus:
    # stimulus represents a perceptual situation where two exact quantities are given

    def __init__(self, a=None, b=None):
        if a is None:
            self.a = randint(1, 101)
        else:
            self.a = a
        if b is None:
            self.b = randint(1, 101)
        else:
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
        try:
            self.interp = interp1d(x, y) #radial basis interpolation?
        except ValueError:
            print("x and y arrays must have at least 2 entries")
            print(x)
            print(y)
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
        # TODO what if the same stimuli?
        responses = [self.response(s) for s in stimuli]
        max_response = max(responses)
        which = [i for i, j in enumerate(responses) if j == max_response]
        return which[0] if len(which) == 1 else None
        # TODO example: responses == [0.0, 0.0]

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


class Perception(Viewable):

    discriminative_threshold = 0.95

    class Error(Error):
        NO_CATEGORY = Error._END_ - 1                   # agent has no categories
        NO_DISCRIMINATION_LOWER_1 = Error._END_ - 2     # agent has categories but is unable to discriminate, lower response for stimulus 1
        NO_DISCRIMINATION_LOWER_2 = Error._END_ - 3     # agent has categories but is unable to discriminate, lower response for stimulus 2
        NO_DIFFERENCE_FOR_CATEGORY = Error._END_ - 4    # agent fails to select topic using category bcs it produces the same responses for both stimuli
        NO_POSITIVE_RESPONSE_1 = Error._END_ - 5        # agent has categories but they return 0 as response for stimulus 1
        NO_POSITIVE_RESPONSE_2 = Error._END_ - 6        # agent has categories but they return 0 as response for stimulus 2
        NO_NOTICEABLE_DIFFERENCE = Error._END_ - 7      # stimuli are indistinguishable for agent perception (jnd)
        _END_ = NO_NOTICEABLE_DIFFERENCE

    def __init__(self):
        self.categories = []

    def discriminate(self, context, topic):
        if not self.categories:
            return None, Perception.Error.NO_CATEGORY

        s1, s2 = context[0], context[1]

        if not Perception.noticeable_difference(s1,s2):
            return None, Perception.Error.NO_NOTICEABLE_DIFFERENCE

        responses1 = [c.response(s1) for c in self.categories]
        responses2 = [c.response(s2) for c in self.categories]
        max1, max2 = max(responses1), max(responses2)
        max_args1 = [i for i, j in enumerate(responses1) if j == max1]
        max_args2 = [i for i, j in enumerate(responses2) if j == max2]

        # TODO discuss
        if max1 == 0:
            return None, Perception.Error.NO_POSITIVE_RESPONSE_1

        if max2 == 0:
            return None, Perception.Error.NO_POSITIVE_RESPONSE_2

        if len(max_args1) > 1 or len(max_args2) > 1:
            raise Exception("Two categories give the same maximal value for stimulus")

        i, j = max_args1[0], max_args2[0]

        if i == j:
            # self.store_ds_result(self.Result.FAILURE)
            return (None, Perception.Error.NO_DISCRIMINATION_LOWER_1) if max1 < max2 else \
                (None, Perception.Error.NO_DISCRIMINATION_LOWER_2)

        #discrimination successful
        return i if topic == 0 else j, Perception.Error.NO_ERROR

    # TODO adhoc implementation of noticeable difference between stimuli
    # TODO doesnt seem to work, try out simulation
    # Stimulus 1: 7 / 75 = 0.093333
    # Stimulus 2: 6 / 84 = 0.071429
    # topic = 2
    # discrimination failure no discrimination
    # Speaker(1) learns topic by adding new category
    @staticmethod
    def noticeable_difference(stimulus1, stimulus2):
        p1 = (stimulus1.a / stimulus1.b)
        p2 = (stimulus2.a / stimulus2.b)
        ds = min(0.3 * p1, 0.3 * p2)
        return abs(p1-p2) > ds

    def plot(self, filename=None, x_left=0, x_right=100, mode=''):
        plt.title("categories")
        ax = plt.gca()
        plt.xscale("symlog")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        plt.yscale("symlog")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        x = linspace(x_left, x_right, 20*(x_right-x_left), False)
        for c in self.categories:
            plt.plot(x, [c.fun(x_0) for x_0 in x], '-', label="%d" % (self.categories.index(c) + 1))
        plt.legend(loc="best")
        plt.savefig(filename)
        plt.close()
