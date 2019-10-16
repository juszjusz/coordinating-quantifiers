from __future__ import division  # force python 3 division in python 2
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

        while not s1.is_noticeably_different_from(s2):
            s1 = self.stimulus_factory._new_stimulus()
            s2 = self.stimulus_factory._new_stimulus()

        return [s1, s2]


class QuotientBasedStimulus:
    # stimulus represents a perceptual situation where two exact quantities are given
    def __init__(self, a, b, sigma):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.real = self.a/self.b

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
        return str({'value': '{}/{} = {}'.format(self.a, self.b, self.a/self.b), 'sigma': self.sigma})

    def is_noticeably_different_from(self, stimulus):
        p1 = (self.a / self.b)
        p2 = (stimulus.a / stimulus.b)
        ds = min(0.3 * p1, 0.3 * p2)
        return abs(p1 - p2) > ds


class StimulusFactory:

    sigma = 0.1
    max = 100

    def __init__(self, x):
        self.x = x
        pass


class QuotientBasedStimulusFactory(StimulusFactory):

    def __init__(self):
        #all_stimuli = [QuotientBasedStimulus(a, b, StimulusFactory.sigma) for b in range(1, StimulusFactory.max) for a in range(1, b + 1)]
        StimulusFactory.__init__(self, np.linspace(0.0, 1.0, 101, endpoint=False))
        self.b_factory = lambda: randint(1, StimulusFactory.max)

    def _new_stimulus(self):
        b = self.b_factory()
        return QuotientBasedStimulus(randint(1, b), b, self.sigma)


class NumericBasedStimulus:

    def __init__(self, value, sigma):
        self.value = value
        self.sigma = self.value * sigma
        self.real = float(value)

    def pdf(self):
        return ProbabilityDensityFunction(self.value - 3.0 * self.sigma,
                                          self.value + 3.0 * self.sigma,
                                          norm(self.value, self.sigma).pdf)

    def __str__(self):
        return str({'value': self.value, 'sigma': self.sigma})

    def is_noticeably_different_from(self, stimulus):
        delta_i1 = 0.1 * self.value  # addition required for a change to be perceived, Weber constant 0.1
        delta_i2 = 0.1 * stimulus.value
        return abs(self.value - stimulus.value) > min(delta_i1, delta_i2)


class NumericBasedStimulusFactory(StimulusFactory):

    def __init__(self):
        all_stimuli = [NumericBasedStimulus(n, StimulusFactory.sigma) for n in range(0, StimulusFactory.max + 1)]
        x = list(set([s.real for s in all_stimuli]))
        x.sort()
        StimulusFactory.__init__(self, x)
        self.a_factory = lambda: randint(0, StimulusFactory.max)

    def _new_stimulus(self):
        return NumericBasedStimulus(self.a_factory(), StimulusFactory.sigma)


class ProbabilityDensityFunction:
    def __init__(self, x_left, x_right, pdf):
        self.x_left = x_left
        self.x_right = x_right
        self.pdf = pdf


stimulus_factory = {
    'numeric': NumericBasedStimulusFactory(),
    'quotient': QuotientBasedStimulusFactory()
}

sf = None

context_factory = {
    'numeric': ContextFactory(stimulus_factory['numeric']),
    'quotient': ContextFactory(stimulus_factory['quotient'])
}
