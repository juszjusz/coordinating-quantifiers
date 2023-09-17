import dataclasses
import os
import time
from fractions import Fraction
from functools import singledispatch, lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, Union, Callable, Any

import h5py
import numpy as np
from scipy.stats import norm

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

    def dot_product_all(self, i: List[Stimulus]):
        # nie wiem czy to dobra nazwa
        pass

    def activation_from_responses(self, response_over_stimuli: List[float]):
        pass


class NumericCalculator(Calculator):

    def __init__(self, stimuli: List[int], support, distribution, reactive_x_reactive):
        self._numeric_to_index = {v: index for index, v in enumerate(stimuli)}
        self._domain = support
        self._reactive_unit_distribution = distribution
        self._reactive_x_reactive = reactive_x_reactive

    def domain(self):
        return self._domain

    def dot_product(self, r1: int, r2: int):
        i1 = self._numeric_to_index[r1]
        i2 = self._numeric_to_index[r2]
        return self._reactive_x_reactive[i1][i2]

    def dot_product_all(self, rs1: List[int]):
        is1 = [self._numeric_to_index[r1] for r1 in rs1]
        return self._reactive_x_reactive[:, is1]

    def pdf(self, r: int):
        i = self._numeric_to_index[r]
        return self._reactive_unit_distribution[i]

    def activation_from_responses(self, response_over_stimuli: List[float]):
        return np.array(response_over_stimuli).astype(bool)

    @staticmethod
    def from_description_with_no_ans(sigma_factor=.3):
        stimuli = tuple([int(x) for x in np.arange(1, 101).astype(int)])
        support = tuple(np.arange(-5.5, 105.5, .01))
        pdf = [norm.pdf(support, loc=s, scale=sigma_factor) for s in stimuli]
        rxr = np.dot(support, np.transpose(support))
        return stimuli, NumericCalculator(stimuli, support, pdf, rxr)

    @staticmethod
    def from_description_with_ans():
        stimuli = tuple([int(x) for x in np.arange(1, 101).astype(int)])
        support = tuple(np.arange(0, 150., .01))
        pdf = [norm.pdf(support, loc=s, scale=s / 10) for s in stimuli]
        rxr = np.dot(pdf, np.transpose(pdf))
        return stimuli, NumericCalculator(stimuli, support, pdf, rxr)

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

    def dot_product_all(self, rs1: List[QuotientStimulus]):
        is1 = [self._quotients_to_index[QuotientCalculator._fraction_index(r1)] for r1 in rs1]
        return self._reactive_x_reactive[:, is1]

    def pdf(self, r: QuotientStimulus):
        i = self._quotients_to_index[QuotientCalculator._fraction_index(r)]
        return self._reactive_unit_distribution[i]

    def activation_from_responses(self, response_over_stimuli: List[float]):
        window_size = 5
        activations = np.array(response_over_stimuli).astype(bool)
        activations = [activations[max(0, i - window_size):min(len(activations), i + window_size)] for i in
                       range(len(activations))]

        middle = np.mean(activations[window_size:-window_size], axis=1)
        head = [np.mean(boundary_activation) for boundary_activation in activations[:window_size]]
        tail = [np.mean(boundary_activation) for boundary_activation in activations[-window_size:]]
        activations = np.concatenate((head, middle, tail))
        return activations > .5

    @staticmethod
    def compute_pdf(support, stimuli_bucket, sigma_factor):
        return [(s, norm.pdf(support, loc=s, scale=sigma_factor)) for s in stimuli_bucket]

    @staticmethod
    def compute_pdf_with_ans(support, stimuli_bucket):
        return [(s, norm.pdf(support, loc=s, scale=s * .1)) for s in stimuli_bucket]

    @staticmethod
    def from_description_with_no_ans(sigma_factor=.003):
        fractions = list(set([Fraction(nom, denom) for denom in range(1, 101) for nom in range(1, denom + 1)]))
        fractions = sorted(fractions)
        stimuli = fractions
        support = tuple(np.arange(0., 2., .001))
        # pdf = [norm.pdf(support, loc=s, scale=sigma_factor) for s in stimuli]

        processes = 12
        bucket_size = int(len(stimuli) / processes)
        stimuli_buckets = [stimuli[i:i + bucket_size] for i in range(0, len(stimuli), bucket_size)]
        with Pool(processes=processes) as pool:
            args = [(support, stimuli_bucket, sigma_factor) for stimuli_bucket in stimuli_buckets]
            pdfs = pool.starmap(QuotientCalculator.compute_pdf, args)
        pdf = [s2pdf for pdf in pdfs for s2pdf in pdf]
        pdf = sorted(pdf, key=lambda s2pdf: s2pdf[0])
        pdf = [pdf for _, pdf in pdf]
        rxr = np.dot(pdf, np.transpose(pdf))
        return stimuli, QuotientCalculator(stimuli, support, pdf, rxr)

    @staticmethod
    def from_description_with_ans():
        fractions = list(set([Fraction(nom, denom) for denom in range(1, 101) for nom in range(1, denom + 1)]))
        fractions = sorted(fractions)
        stimuli = fractions
        support = tuple(np.arange(0., 2.3, .001))

        processes = 12
        bucket_size = int(len(stimuli) / processes)
        stimuli_buckets = [stimuli[i:i + bucket_size] for i in range(0, len(stimuli), bucket_size)]
        with Pool(processes=processes) as pool:
            args = [(support, stimuli_bucket) for stimuli_bucket in stimuli_buckets]
            pdfs = pool.starmap(QuotientCalculator.compute_pdf_with_ans, args)
        pdf = [s2pdf for pdf in pdfs for s2pdf in pdf]
        pdf = sorted(pdf, key=lambda s2pdf: s2pdf[0])
        pdf = [pdf for _, pdf in pdf]
        rxr = np.dot(pdf, np.transpose(pdf))
        return stimuli, QuotientCalculator(stimuli, support, pdf, rxr)

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
        return NumericCalculator.from_description_with_ans()
    if stimuli_type == 'numeric' and not with_ans:
        return NumericCalculator.from_description_with_no_ans()
