from __future__ import division  # force python 3 division in python 2

from fractions import Fraction
from random import randint

class AbstractStimulus:
    def is_noticeably_different_from(self, other):
        raise NotImplementedError


class ContextFactory:
    def __init__(self, stimulus_factory):
        self.new_stimulus = stimulus_factory

    def __call__(self):
        s1 = self.new_stimulus()
        s2 = self.new_stimulus()

        while not s1.is_noticeably_different_from(s2):
            s1 = self.new_stimulus()
            s2 = self.new_stimulus()

        return [s1, s2]


class NumericBasedStimulusFactory:
    def __init__(self, max):
        self.__max = max

    def __call__(self):
        n = randint(0, self.__max)
        return NumericBasedStimulus(n)

class QuotientBasedStimulusFactory:
    def __init__(self, stimulus_list, max):
        self.stimulus_list_arr = stimulus_list
        self.stimulus_list = [list(self.stimulus_list_arr[i]) for i in range(0, len(self.stimulus_list_arr))]
        self.max = max
        self.filtered_stimulus_list = [QuotientBasedStimulus(self.stimulus_list.index(s), Fraction(s[0], s[1])) for s in self.stimulus_list if s[1] <= self.max]

    def get_stimuli(self):
        return self.filtered_stimulus_list

    def __call__(self):
        k = randint(1, self.max)
        n = randint(1, k)
        f = Fraction(n,k)
        pair = [f.numerator, f.denominator]
        index = self.stimulus_list.index(pair)
        return QuotientBasedStimulus(index, f)
        #index = randint(0, len(self.stimulus_list) - 1)
        #n, k = self.stimulus_list[index]
        #return QuotientBasedStimulus(index, Fraction(n, k))

class NumericBasedStimulus(AbstractStimulus):
    def __init__(self, n):
        self.__n = n

    def __str__(self):
        return str(self.__n)

    def is_noticeably_different_from(self, other):
        return True

class QuotientBasedStimulus(AbstractStimulus):
    def __init__(self, index, nk):
        self.index = index
        self.__nk = nk

    def __str__(self):
        return str(self.__nk) + ' = ' + str(float(self.__nk))

    def is_noticeably_different_from(self, other):
        ds = 0.3 * min(self.__nk, other.__nk)
        return abs(self.__nk - other.__nk) > ds

stimulus_factory = None