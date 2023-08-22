from __future__ import division  # force python 3 division in python 2

from typing import Tuple, Any

import numpy as np

from v2.calculator import Calculator, NewAbstractStimulus


class NewCategory:
    def __init__(self, category_id: int):
        self.category_id = category_id
        self.is_active = True
        self._weights = []
        self._reactive_units = []

    def __hash__(self):
        return self.category_id

    def __eq__(self, o) -> bool:
        if not isinstance(o, NewCategory):
            return False

        return self.category_id == o.category_id

    def response(self, stimulus: NewAbstractStimulus, calculator: Calculator):
        return sum([weight * calculator.dot_product(ru_value, stimulus.value()) for weight, ru_value in
                    zip(self._weights, self._reactive_units)])

    def add_reactive_unit(self, stimulus: NewAbstractStimulus, weight=0.5):
        self._weights.append(weight)
        self._reactive_units.append(stimulus.value())

    def select(self, context: Tuple[NewAbstractStimulus, NewAbstractStimulus], calculator: Calculator) -> int or None:
        s1, s2 = context
        r1, r2 = self.response(s1, calculator), self.response(s2, calculator)
        if r1 == r2:
            return None
        else:
            return np.argmax([r1, r2])

    def reinforce(self, stimulus: NewAbstractStimulus, beta, calculator: Calculator):
        self._weights = [weight + beta * calculator.dot_product(ru, stimulus.value()) for weight, ru in
                         zip(self._weights, self._reactive_units)]

    def decrement_weights(self, alpha):
        self._weights = [weight - alpha * weight for weight in self._weights]

    def max_weigth(self):
        return max(self._weights)

    def discretized_distribution(self, calculator: Calculator):
        return self.__apply_fun_to_coordinates(lambda x: np.sum(x, axis=0), calculator)

    def union(self, calculator: Calculator):
        return self.__apply_fun_to_coordinates(lambda x: np.max(x, axis=0), calculator)

    # Given values f(x0),f(x1),...,f(xn); g(x0),g(x1),...,g(xn) for functions f, g defined on points x0 < x1 < ... < xn
    # @__apply_fun_to_coordinates results in FUN(f(x0),g(x0)),FUN(f(x1),g(x1)),...,FUN(f(xn),g(xn))
    # Implementation is defined on family of functions from (REACTIVE_UNIT_DIST[.]).
    def __apply_fun_to_coordinates(self, FUN, calculator: Calculator):
        return FUN([weight * calculator.pdf(ru) for weight, ru in
                    zip(self._weights, self._reactive_units)])

    # def show(self):
    #     DOMAIN = inmem['DOMAIN']
    #     plt.plot(DOMAIN, self.discretized_distribution(), 'o', DOMAIN, self.discretized_distribution(), '--')
    #     plt.legend(['data', 'cubic'], loc='best')
    #     plt.show()

    def deactivate(self):
        self.is_active = False
