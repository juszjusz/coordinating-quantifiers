from __future__ import division  # force python 3 division in python 2

from fractions import Fraction
from random import randint

from inmemory_calculus import inmem

class AbstractStimulus:
    def is_noticeably_different_from(self, other_stimulus):
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

class QuotientBasedStimulusFactory:
    def __init__(self, stimulus_list, max):
        self.stimulus_list_arr = stimulus_list
        self.stimulus_list = [list(self.stimulus_list_arr[i]) for i in range(0, len(self.stimulus_list_arr))]
        #print(list(self.stimulus_list))
        self.max = max

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


class QuotientBasedStimulus(AbstractStimulus):
    def __init__(self, index, nk):
        self.index = index
        self.__nk = nk

    def __str__(self):
        return str(self.__nk)

    def is_noticeably_different_from(self, other):
        ds = 0.3 * min(self.__nk, other.__nk)
        return abs(self.__nk - other.__nk) > ds

