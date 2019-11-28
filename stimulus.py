from __future__ import division  # force python 3 division in python 2

from fractions import Fraction
from random import randint

from inmemory_calculus import NK_LIST


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
    def __init__(self, nk_list=NK_LIST):
        self.nk_list = nk_list

    def __call__(self):
        index = randint(0, len(self.nk_list) - 1)
        n, k = self.nk_list[index]
        return QuotientBasedStimulus(index, Fraction(n, k))


class QuotientBasedStimulus(AbstractStimulus):
    def __init__(self, index, nk):
        self.index = index
        self.__nk = nk

    def is_noticeably_different_from(self, other_stimulus):
        return True
