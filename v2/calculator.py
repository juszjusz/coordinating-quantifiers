import os
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple, Callable

import h5py
import numpy as np


class NewAbstractStimulusFactory:
    def __init__(self, list_of_stimuli: List, random_choice):
        self._list_of_stimuli = list_of_stimuli
        self._random_choice = random_choice

    def __call__(self):
        return self._random_choice(self._list_of_stimuli)

    def get_all_stimuli(self) -> List:
        return self._list_of_stimuli


class NewAbstractStimulus:
    def value(self):
        pass

    def is_noticeably_different_from(self, other) -> bool:
        pass

    def __repr__(self):
        return str(self.value())


class NewNumericStimulus(NewAbstractStimulus):
    def __init__(self, value: int):
        self._value = value

    def value(self):
        return self._value

    def is_noticeably_different_from(self, other: NewAbstractStimulus):
        ds = 0.3 * self.value()
        return abs(self.value() - other.value()) > ds


class NewQuotientStimulus(NewAbstractStimulus):
    def __init__(self, value: Tuple[int, int]):
        self._value = value

    def value(self):
        return self._value

    def is_noticeably_different_from(self, other: NewAbstractStimulus):
        f1 = Fraction(*self.value())
        f2 = Fraction(*other.value())
        ds = 0.3 * f1
        return abs(f1 - f2) > ds


class NewContextFactory:
    def __init__(self, stimulus_factory: Callable[[], NewAbstractStimulus]):
        self._new_stimulus = stimulus_factory

    def __call__(self):
        s1 = self._new_stimulus()
        s2 = self._new_stimulus()

        while not s1.is_noticeably_different_from(s2):
            s1 = self._new_stimulus()
            s2 = self._new_stimulus()

        return [s1, s2]


def read_h5_data(data_path, dataset_key=u'Dataset1'):
    with h5py.File(data_path, 'r') as py_file:
        return py_file[dataset_key][:]


class Calculator:
    def pdf(self, stimulus):
        pass

    def dot_product(self, i, j):
        # nie wiem czy to dobra nazwa
        pass

    def values(self) -> List:
        pass

    def stimuli(self) -> List[NewNumericStimulus]:
        pass

    def context_factory(self, pick_element):
        stimuli = self.stimuli()

        def new_context():
            s1 = pick_element(stimuli)
            s2 = pick_element(stimuli)
            while not s1.is_noticeably_different_from(s2):
                s1 = pick_element(stimuli)
                s2 = pick_element(stimuli)

            return [s1, s2]

        return new_context


class NumericCalculator(Calculator):

    def __init__(self, values, support, distribution, reactive_x_reactive, sigma=.01):
        self._numerics = values
        self._numeric_to_index = {v: index for index, v in enumerate(values)}
        self._domain = support
        self._reactive_unit_distribution = distribution
        self._reactive_x_reactive = reactive_x_reactive
        self._sigma = sigma

    def values(self) -> List[int]:
        return self._numerics.tolist()

    def stimuli(self) -> List[NewNumericStimulus]:
        return [NewNumericStimulus(num) for num in self.values()]

    # ???
    def dot_product(self, r1: int, r2: int):
        i1 = self._numeric_to_index[r1]
        i2 = self._numeric_to_index[r2]
        return self._reactive_x_reactive[i1][i2]

    def pdf(self, r: int):
        i = self._numeric_to_index[r]
        return self._reactive_unit_distribution[i]

    @staticmethod
    def from_description(max_inclusive=100, sigma_factor=.1):
        values = np.arange(1, max_inclusive + 1)
        support = None
        distribution = None
        reactive_x_reactive = None
        return NumericCalculator(values, support, distribution, reactive_x_reactive, sigma_factor)

    @staticmethod
    def load_from_file(path='../inmemory_calculus/numeric'):
        root_path = Path(os.path.abspath(path))

        reactive_unit_distribution = read_h5_data(data_path=root_path.joinpath('R.h5'))
        if not isinstance(reactive_unit_distribution, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(reactive_unit_distribution)))

        reactive_x_reactive = read_h5_data(root_path.joinpath('RxR.h5'))
        if not isinstance(reactive_x_reactive, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(reactive_x_reactive)))

        domain = read_h5_data(root_path.joinpath('domain.h5'))
        if not isinstance(domain, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(domain)))

        stimulus_list = np.arange(1, len(reactive_unit_distribution) + 1)

        # VALIDATE loaded data shapes:
        if not stimulus_list.shape[0] == reactive_unit_distribution.shape[0] == reactive_x_reactive.shape[0] == \
               reactive_x_reactive.shape[1]:
            raise ValueError()
        if not reactive_unit_distribution.shape[1] == domain.shape[0]:
            raise ValueError()

        return NumericCalculator(stimulus_list, domain, reactive_unit_distribution, reactive_x_reactive)


class QuotientCalculator(Calculator):

    def __init__(self, values: List[Tuple[int, int]], support, distribution, reactive_x_reactive, sigma=.01):
        self._quotients = values
        self._quotients_to_index = {v: index for index, v in enumerate(values)}
        self._domain = support
        self._reactive_unit_distribution = distribution
        self._reactive_x_reactive = reactive_x_reactive
        self._sigma = sigma

    def values(self) -> List:
        return self._quotients

    def stimuli(self) -> List[NewQuotientStimulus]:
        return [NewQuotientStimulus((nominator, denominator)) for nominator, denominator in self.values()]

    # ???
    def dot_product(self, r1: Tuple[int, int], r2: Tuple[int, int]):
        i1 = self._quotients_to_index[r1]
        i2 = self._quotients_to_index[r2]
        return self._reactive_x_reactive[i1][i2]

    def pdf(self, r: Tuple[int, int]):
        i = self._quotients_to_index[r]
        return self._reactive_unit_distribution[i]

    @staticmethod
    def from_description():
        pass

    @staticmethod
    def load_from_file(path='../inmemory_calculus/quotient'):
        root_path = Path(os.path.abspath(path))

        reactive_unit_distribution = read_h5_data(data_path=root_path.joinpath('R.h5'))
        if not isinstance(reactive_unit_distribution, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(reactive_unit_distribution)))

        reactive_x_reactive = read_h5_data(root_path.joinpath('RxR.h5'))
        if not isinstance(reactive_x_reactive, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(reactive_x_reactive)))

        domain = read_h5_data(root_path.joinpath('domain.h5'))
        if not isinstance(domain, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(domain)))

        # reduced fractions n/k where n < k and k <= 100; reduced def: n, k are relatively prime integers
        stimulus_list = read_h5_data(root_path.joinpath('nklist.h5'))
        # invoke int(.) for serialization reasons (int64 found here is not json serializable)
        stimulus_list = [(int(nom), int(denom)) for nom, denom in stimulus_list]

        # VALIDATE loaded data shapes:
        if not len(stimulus_list) == reactive_unit_distribution.shape[0] == reactive_x_reactive.shape[0] == \
               reactive_x_reactive.shape[1]:
            raise ValueError()
        if not reactive_unit_distribution.shape[1] == domain.shape[0]:
            raise ValueError()

        return QuotientCalculator(stimulus_list, domain, reactive_unit_distribution, reactive_x_reactive)


if __name__ == '__main__':
    x = QuotientCalculator.load_from_file()
    # 'C:\\Users\\juszynski\\Desktop\\wspace\\coordinating-quantifiers\\inmemory_calculus\\quotient')
    print(x)
