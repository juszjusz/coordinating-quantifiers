from __future__ import division  # force python 3 division in python 2

from copy import copy
from typing import Tuple

import matplotlib.pyplot as plt

from guessing_game_exceptions import NO_NOTICEABLE_DIFFERENCE, NO_CATEGORY, NO_DISCRIMINATION
from collections import deque
from inmemory_calculus import inmem

from numpy import max, sum
from numpy.random import RandomState

from stimulus import AbstractStimulus


class NewCategory:
    def __init__(self, category_id: int, seed: int):
        self.category_id = category_id
        self.is_active = True
        self.__weights = []
        self.__reactive_indicies = []
        self.__random = RandomState(seed)

    def __hash__(self):
        return self.category_id


    def response(self, stimulus, REACTIVE_X_REACTIVE=None):
        if REACTIVE_X_REACTIVE is None:
            REACTIVE_X_REACTIVE = inmem['REACTIVE_X_REACTIVE']
        return sum([weight * REACTIVE_X_REACTIVE[ru_index][stimulus.index] for weight, ru_index in
                    zip(self.__weights, self.__reactive_indicies)])

    def add_reactive_unit(self, stimulus, weight=0.5):
        self.__weights.append(weight)
        self.__reactive_indicies.append(stimulus.index)

    def select(self, context: Tuple[AbstractStimulus]) -> int:
        # TODO what if the same stimuli?
        responses = [self.response(s) for s in context]
        max_response = max(responses)
        which = [i for i, j in enumerate(responses) if j == max_response]
        return which[0] if len(which) == 1 else self.__random.choice([0, 1])
        # TODO example: responses == [0.0, 0.0]

    def reinforce(self, stimulus, beta, REACTIVE_X_REACTIVE=None):
        if REACTIVE_X_REACTIVE is None:
            REACTIVE_X_REACTIVE = inmem['REACTIVE_X_REACTIVE']
        self.__weights = [weigth + beta * REACTIVE_X_REACTIVE[ru_index][stimulus.index] for weigth, ru_index in
                          zip(self.__weights, self.__reactive_indicies)]

    def decrement_weights(self, alpha):
        self.__weights = [weight - alpha * weight for weight in self.__weights]

    def max_weigth(self):
        return max(self.__weights)

    def discretized_distribution(self, REACTIVE_UNIT_DIST=None):
        return self.__apply_fun_to_coordinates(lambda x: sum(x, axis=0), REACTIVE_UNIT_DIST)

    def union(self, REACTIVE_UNIT_DIST=None):
        return self.__apply_fun_to_coordinates(lambda x: max(x, axis=0), REACTIVE_UNIT_DIST)

    # Given values f(x0),f(x1),...,f(xn); g(x0),g(x1),...,g(xn) for functions f, g defined on points x0 < x1 < ... < xn
    # @__apply_fun_to_coordinates results in FUN(f(x0),g(x0)),FUN(f(x1),g(x1)),...,FUN(f(xn),g(xn))
    # Implementation is defined on family of functions from (REACTIVE_UNIT_DIST[.]).
    def __apply_fun_to_coordinates(self, FUN, REACTIVE_UNIT_DIST=None):
        if REACTIVE_UNIT_DIST is None:
            REACTIVE_UNIT_DIST = inmem['REACTIVE_UNIT_DIST']
        return FUN([weight * REACTIVE_UNIT_DIST[ru_index] for weight, ru_index in
                    zip(self.__weights, self.__reactive_indicies)])

    def show(self):
        DOMAIN = inmem['DOMAIN']
        plt.plot(DOMAIN, self.discretized_distribution(), 'o', DOMAIN, self.discretized_distribution(), '--')
        plt.legend(['data', 'cubic'], loc='best')
        plt.show()

    def deactivate(self):
        self.is_active = False
