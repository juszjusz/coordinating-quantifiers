from __future__ import division  # force python 3 division in python 2

import matplotlib.pyplot as plt

from guessing_game_exceptions import NO_POSITIVE_RESPONSE_1, NO_POSITIVE_RESPONSE_2, NO_DISCRIMINATION_LOWER_1, \
    NO_DISCRIMINATION_LOWER_2, NO_NOTICEABLE_DIFFERENCE, NO_CATEGORY, NO_DISCRIMINATION
from collections import deque
from inmemory_calculus import inmem
import numpy as np
from random import choice

class Category:
    def __init__(self, id):
        self.id = id
        self.__weights = []
        self.__reactive_indicies = []

    def response(self, stimulus, REACTIVE_X_REACTIVE=None):
        if REACTIVE_X_REACTIVE is None:
            REACTIVE_X_REACTIVE = inmem['REACTIVE_X_REACTIVE']
        return sum([weight * REACTIVE_X_REACTIVE[ru_index][stimulus.index] for weight, ru_index in zip(self.__weights, self.__reactive_indicies)])

    def add_reactive_unit(self, stimulus, weight=0.5):
        self.__weights.append(weight)
        self.__reactive_indicies.append(stimulus.index)

    def select(self, stimuli):
        # TODO what if the same stimuli?
        responses = [self.response(s) for s in stimuli]
        max_response = max(responses)
        which = [i for i, j in enumerate(responses) if j == max_response]
        return which[0] if len(which) == 1 else choice([0,1])
        # TODO example: responses == [0.0, 0.0]

    def reinforce(self, stimulus, beta, REACTIVE_X_REACTIVE=None):
        if REACTIVE_X_REACTIVE is None:
            REACTIVE_X_REACTIVE = inmem['REACTIVE_X_REACTIVE']
        self.__weights = [weigth + beta * REACTIVE_X_REACTIVE[ru_index][stimulus.index] for weigth, ru_index in zip(self.__weights, self.__reactive_indicies)]

    def decrement_weights(self, alpha):
        self.__weights = [weigth - alpha * weigth for weigth in self.__weights]

    def max_weigth(self):
        return max(self.__weights)

    def discretized_distribution(self, REACTIVE_UNIT_DIST=None):
        return self.__apply_fun_to_coordinates(lambda x: np.sum(x, axis=0), REACTIVE_UNIT_DIST)

    def union(self, REACTIVE_UNIT_DIST=None):
        return self.__apply_fun_to_coordinates(lambda x: np.max(x, axis=0), REACTIVE_UNIT_DIST)

    # Given values f(x0),f(x1),...,f(xn); g(x0),g(x1),...,g(xn) for functions f, g defined on points x0 < x1 < ... < xn
    # @__apply_fun_to_coordinates results in FUN(f(x0),g(x0)),FUN(f(x1),g(x1)),...,FUN(f(xn),g(xn))
    # Implementation is defined on family of functions from (REACTIVE_UNIT_DIST[.]).
    def __apply_fun_to_coordinates(self, FUN, REACTIVE_UNIT_DIST=None):
        if REACTIVE_UNIT_DIST is None:
            REACTIVE_UNIT_DIST = inmem['REACTIVE_UNIT_DIST']
        return FUN([weight * REACTIVE_UNIT_DIST[ru_index] for weight, ru_index in zip(self.__weights, self.__reactive_indicies)])

    def show(self):
        DOMAIN = inmem['DOMAIN']
        plt.plot(DOMAIN, self.discretized_distribution(), 'o', DOMAIN, self.discretized_distribution(), '--')
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

    def get_best_matching_category(self, stimulus):
        responses = [c.response(stimulus) for c in self.categories]
        try:
            max_resp = max(responses)
        except ValueError:
            return None
        max_args = [i for i, j in enumerate(responses) if j == max_resp]
        return max_args[0]

    def discriminate(self, context, topic):
        if not self.categories:
            # self.store_ds_result(Perception.Result.FAILURE)
            raise NO_CATEGORY

        s1, s2 = context[0], context[1]

        # TODO do wywalnie prawdopodobnie, ze wzgledu na sposob generowania kontekstow
        if not s1.is_noticeably_different_from(s2):
            # self.store_ds_result(Perception.Result.FAILURE)
            raise NO_NOTICEABLE_DIFFERENCE

        i = self.get_best_matching_category(s1)
        j = self.get_best_matching_category(s2)
        # TODO discuss
        #if max1 == 0.0:
        #    # self.store_ds_result(Perception.Result.FAILURE)
        #    raise NO_POSITIVE_RESPONSE_1

        #if max2 == 0.0:
        #    # self.store_ds_result(Perception.Result.FAILURE)
        #    raise NO_POSITIVE_RESPONSE_2

        #if len(max_args1) > 1 or len(max_args2) > 1:
        #    raise Exception("Two categories give the same maximal value for stimulus")

        if i == j:
            raise NO_DISCRIMINATION
            # self.store_ds_result(Perception.Result.FAILURE)
            #raise NO_DISCRIMINATION_LOWER_1(i) if max1 < max2 else \
            #    NO_DISCRIMINATION_LOWER_2(i)

        # discrimination successful
        # self.store_ds_result(Perception.Result.SUCCESS)
        return self.categories[i] if topic == 0 else self.categories[j]