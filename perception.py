from __future__ import division  # force python 3 division in python 2

import logging
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

from guessing_game_exceptions import NO_POSITIVE_RESPONSE_1, NO_POSITIVE_RESPONSE_2, NO_DISCRIMINATION_LOWER_1, \
    NO_DISCRIMINATION_LOWER_2, NO_NOTICEABLE_DIFFERENCE, NO_CATEGORY
from collections import deque

class Category:

    def __init__(self, id, rxr, ri):
        self.id = id
        self.weights = []
        self.reactive_indicies = []
        self.rxr = rxr
        self.ri = ri
        self.x_left = float("inf")
        self.x_right = float("-inf")

    def response(self, stimulus):
        return sum([w * self.rxr[ru_index][stimulus.index] for ru_index, w in zip(self.reactive_indicies, self.weights)])

    def add_reactive_unit(self, stimulus, weight=0.5):
        self.weights.append(weight)
        self.reactive_indicies.append(stimulus.index)

    def select(self, stimuli):
        # TODO what if the same stimuli?
        responses = [self.response(s) for s in stimuli]
        max_response = max(responses)
        which = [i for i, j in enumerate(responses) if j == max_response]
        return which[0] if len(which) == 1 else None
        # TODO example: responses == [0.0, 0.0]

    def reinforce(self, stimulus, beta):
        self.weights = [w + beta * self.rxr[ru_index][stimulus.index] for ru_index, w in zip(self.reactive_indicies, self.weights)]

    def area(self):
        x_delta = .001
        return x_delta * sum([sum([v * w for v in self.ri[index]]) for index, w in zip(self.reactive_indicies, self.weights)])

    def show(self):
        x = np.linspace(self.x_left, self.x_right, num=100)
        plt.plot(x, self.fun(x), 'o', x, self.fun(x), '--')
        plt.legend(['data', 'cubic'], loc='best')
        plt.show()


class Perception:
    class Result:
        SUCCESS = 1
        FAILURE = 0

    def __init__(self):
        self.categories = []
        self.ds_scores = deque([0])
        self._id_ = 0
        self.discriminative_success = 0.0

    def get_cat_id(self):
        self._id_ = self._id_ + 1
        return self._id_ - 1

    def store_ds_result(self, result):
        if len(self.ds_scores) == 50:
            self.ds_scores.rotate(-1)
            self.ds_scores[-1] = result
        else:
            self.ds_scores.append(result)
        self.discriminative_success = sum(self.ds_scores) / len(self.ds_scores)

    def switch_ds_result(self):
        self.ds_scores[-1] = 1 - self.ds_scores[-1]
        self.discriminative_success = sum(self.ds_scores) / len(self.ds_scores)

    def discriminate(self, context, topic):
        if not self.categories:
            # self.store_ds_result(Perception.Result.FAILURE)
            raise NO_CATEGORY

        s1, s2 = context[0], context[1]

        # TODO do wywalnie prawdopodobnie, ze wzgledu na sposob generowania kontekstow
        if not s1.is_noticeably_different_from(s2):
            # self.store_ds_result(Perception.Result.FAILURE)
            raise NO_NOTICEABLE_DIFFERENCE

        responses1 = [c.response(s1) for c in self.categories]
        responses2 = [c.response(s2) for c in self.categories]
        max1, max2 = max(responses1), max(responses2)
        max_args1 = [i for i, j in enumerate(responses1) if j == max1]
        max_args2 = [i for i, j in enumerate(responses2) if j == max2]

        # TODO discuss
        if max1 == 0.0:
            # self.store_ds_result(Perception.Result.FAILURE)
            raise NO_POSITIVE_RESPONSE_1

        if max2 == 0.0:
            # self.store_ds_result(Perception.Result.FAILURE)
            raise NO_POSITIVE_RESPONSE_2

        if len(max_args1) > 1 or len(max_args2) > 1:
            raise Exception("Two categories give the same maximal value for stimulus")

        i, j = max_args1[0], max_args2[0]

        if i == j:
            # self.store_ds_result(Perception.Result.FAILURE)
            raise NO_DISCRIMINATION_LOWER_1(i) if max1 < max2 else \
                NO_DISCRIMINATION_LOWER_2(i)

        # discrimination successful
        # self.store_ds_result(Perception.Result.SUCCESS)
        return self.categories[i] if topic == 0 else self.categories[j]