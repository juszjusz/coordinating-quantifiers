import os
from fractions import Fraction
from functools import singledispatch, lru_cache
from pathlib import Path
from typing import List, Tuple, Union, Callable, Any

import h5py
import numpy as np

NumericStimulus = int
QuotientStimulus = Fraction
Stimulus = Union[NumericStimulus, QuotientStimulus]
NumericStimulusContext = Tuple[NumericStimulus, NumericStimulus]
QuotientStimulusContext = Tuple[QuotientStimulus, QuotientStimulus]
StimulusContext = Union[NumericStimulusContext, QuotientStimulusContext]


@singledispatch
def is_noticeably_different_from(arg0, arg1):
    pass


@is_noticeably_different_from.register(int)
@is_noticeably_different_from.register(int)
def _(i, j):
    ds = 0.3 * i
    return abs(i - j) > ds


@is_noticeably_different_from.register(Fraction)
@is_noticeably_different_from.register(Fraction)
def _(f1, f2):
    ds = 0.3 * f1
    return abs(f1 - f2) > ds


def context_factory(stimuli: List[Stimulus], pick_element: Callable[[List[Any]], Any]):
    def new_context() -> StimulusContext:
        s1 = pick_element(stimuli)
        s2 = pick_element(stimuli)
        while not is_noticeably_different_from(s1, s2):
            s1 = pick_element(stimuli)
            s2 = pick_element(stimuli)

        return s1, s2

    return new_context


def read_h5_data(data_path, dataset_key=u'Dataset1'):
    with h5py.File(data_path, 'r') as py_file:
        return py_file[dataset_key][:]


class Calculator:
    def domain(self):
        pass

    def pdf(self, stimulus):
        pass

    def dot_product(self, i: Stimulus, j: Stimulus):
        # nie wiem czy to dobra nazwa
        pass


class NumericCalculator(Calculator):

    def __init__(self, stimuli: List[int], support, distribution, reactive_x_reactive, sigma=.01):
        self._numeric_to_index = {v: index for index, v in enumerate(stimuli)}
        self._domain = support
        self._reactive_unit_distribution = distribution
        self._reactive_x_reactive = reactive_x_reactive
        self._sigma = sigma

    def domain(self):
        return self._domain

    def dot_product(self, r1: int, r2: int):
        i1 = self._numeric_to_index[r1]
        i2 = self._numeric_to_index[r2]
        return self._reactive_x_reactive[i1][i2]

    def pdf(self, r: int):
        i = self._numeric_to_index[r]
        return self._reactive_unit_distribution[i]

    @staticmethod
    def from_description(max_inclusive=100, sigma_factor=.1):
        pass

    @staticmethod
    def load_from_file_with_ans():
        return NumericCalculator.load_from_file('../inmemory_calculus_ans/numeric')

    @staticmethod
    def load_from_file_with_no_ans():
        return NumericCalculator.load_from_file('../inmemory_calculus_no_ans/numeric')

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

        stimuli = [*range(1, len(reactive_unit_distribution) + 1)]

        # VALIDATE loaded data shapes:
        if not len(stimuli) == reactive_unit_distribution.shape[0] == reactive_x_reactive.shape[0] == \
               reactive_x_reactive.shape[1]:
            raise ValueError()
        if not reactive_unit_distribution.shape[1] == domain.shape[0]:
            raise ValueError()

        calculator = NumericCalculator(stimuli, domain, reactive_unit_distribution, reactive_x_reactive)

        return stimuli, calculator


class QuotientCalculator(Calculator):

    def __init__(self, stimuli: List[QuotientStimulus], support, distribution, reactive_x_reactive):
        self._quotients_to_index = {QuotientCalculator._fraction_index(v): index for index, v in enumerate(stimuli)}
        self._domain = support
        self._reactive_unit_distribution = distribution
        self._reactive_x_reactive = reactive_x_reactive

    @staticmethod
    def _fraction_index(f: QuotientStimulus) -> Tuple[int, int]:
        return f.numerator, f.denominator

    def domain(self):
        return self._domain

    def dot_product(self, r1: QuotientStimulus, r2: QuotientStimulus):
        i1 = self._quotients_to_index[QuotientCalculator._fraction_index(r1)]
        i2 = self._quotients_to_index[QuotientCalculator._fraction_index(r2)]
        return self._reactive_x_reactive[i1][i2]

    def pdf(self, r: QuotientStimulus):
        i = self._quotients_to_index[QuotientCalculator._fraction_index(r)]
        return self._reactive_unit_distribution[i]

    @staticmethod
    def from_description():
        pass

    @staticmethod
    def load_from_file_with_ans():
        return QuotientCalculator.load_from_file('../inmemory_calculus_ans/quotient')

    @staticmethod
    def load_from_file_with_no_ans():
        return QuotientCalculator.load_from_file('../inmemory_calculus_no_ans/quotient')

    @staticmethod
    def load_from_file(path='../inmemory_calculus/quotient') -> Tuple[List[QuotientStimulus], Calculator]:
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
        stimuli = read_h5_data(root_path.joinpath('nklist.h5'))
        # invoke int(.) for serialization reasons (int64 is not json serializable)
        stimuli = [Fraction(int(nom), int(denom)) for nom, denom in stimuli]

        # VALIDATE loaded data shapes:
        if not len(stimuli) == reactive_unit_distribution.shape[0] == reactive_x_reactive.shape[0] == \
               reactive_x_reactive.shape[1]:
            raise ValueError()
        if not reactive_unit_distribution.shape[1] == domain.shape[0]:
            raise ValueError()

        return stimuli, QuotientCalculator(stimuli, domain, reactive_unit_distribution, reactive_x_reactive)


def load_stimuli_and_calculator(stimuli_type, with_ans=True):
    assert stimuli_type in {'quotient', 'numeric'}
    if stimuli_type == 'quotient' and with_ans:
        return QuotientCalculator.load_from_file_with_ans()
    if stimuli_type == 'quotient' and not with_ans:
        return QuotientCalculator.load_from_file_with_no_ans()
    if stimuli_type == 'numeric' and with_ans:
        return NumericCalculator.load_from_file_with_ans()
    if stimuli_type == 'numeric' and not with_ans:
        return NumericCalculator.load_from_file_with_no_ans()

