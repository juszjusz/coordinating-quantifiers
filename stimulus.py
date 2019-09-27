from random import randint
from scipy.stats import norm
import numpy as np

from math_utils import interpolate


class ContextFactory:
    def __init__(self, stimulus_factory):
        self.stimulus_factory = stimulus_factory

    def __call__(self, *args, **kwargs):
        s1 = self.stimulus_factory._new_stimulus()
        s2 = self.stimulus_factory._new_stimulus()

        while not self.stimulus_factory._is_noticeable_difference(s1, s2):
            s1 = self.stimulus_factory._new_stimulus()
            s2 = self.stimulus_factory._new_stimulus()

        return [s1, s2]


class QuotientBasedStimulusFactory():
    def __init__(self, a_factory=lambda: randint(1, 101), b_factory=lambda: randint(1, 101), sigma=.1):
        self.a_factory = a_factory
        self.b_factory = b_factory
        self.sigma = sigma

    def _new_stimulus(self):
        a = self.a_factory()
        b = self.b_factory()
        return QuotientBasedStimulus(a, b, self.sigma)

    def _is_noticeable_difference(self, stimulus1, stimulus2):
        p1 = (stimulus1.a / stimulus1.b)
        p2 = (stimulus2.a / stimulus2.b)
        ds = min(0.3 * p1, 0.3 * p2)
        return abs(p1 - p2) > ds


class QuotientBasedStimulus:
    # stimulus represents a perceptual situation where two exact quantities are given
    def __init__(self, a, b, sigma):
        self.a = a
        self.b = b
        self.sigma = sigma

    def pdf(self):
        # TODO consider different interpolation methods
        a_samples = self.__sample(self.a)
        b_samples = self.__sample(self.b)
        r = a_samples / b_samples
        ratio_samples = r.tolist()
        y, bin_edges = np.histogram(ratio_samples, bins="auto", density=True)
        x = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(0, len(bin_edges) - 1)]

        return ProbabilityDensityFunction(x[0], x[-1], interpolate(x, y))

    def __sample(self, quantity, sample_size=10000):
        return np.array(np.random.normal(quantity, self.sigma * quantity, sample_size), dtype=np.float)

    def __str__(self):
        return str({'value': '{}/{}'.format(self.a, self.b), 'sigma': self.sigma})


class NumericBasedStimulus:
    def __init__(self, value, sigma):
        self.value = value
        self.sigma = sigma

    def pdf(self):
        return ProbabilityDensityFunction(None, None, norm(self.value, self.sigma).pdf)

    def __str__(self):
        return str({'value': self.value, 'sigma': self.sigma})


class NumericBasedStimulusFactory():
    def __init__(self, a_factory=lambda: randint(1, 101), sigma=.1):
        self.a_factory = a_factory
        self.sigma = sigma

    def _new_stimulus(self):
        return NumericBasedStimulus(self.a_factory(), self.sigma)

    def _is_noticeable_difference(self, s1, s2):
        return abs(s1.value - s2.value)


class ProbabilityDensityFunction:
    def __init__(self, x_left, x_right, pdf):
        self.x_left = x_left
        self.x_right = x_right
        self.pdf = pdf


context_factory = {
    'numeric': ContextFactory(NumericBasedStimulusFactory()),
    'quotient': ContextFactory(QuotientBasedStimulusFactory())
}
