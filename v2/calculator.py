import dataclasses
import os
from fractions import Fraction
from functools import singledispatch
from pathlib import Path
from typing import List, Tuple, Union, Callable, Any, Dict

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


def calculate_normal_pdfs(support: List[float], means: List[Tuple], sigmas: List[float]):
    assert len(means) == len(sigmas), 'expects means & sigmas to be of equal sizes'
    support_size = len(support)
    size = len(means)
    total_size = support_size * size

    supports = np.tile(support, size).reshape((total_size,)).astype(np.float32)
    repeated_means = np.repeat(means, support_size).reshape((total_size,)).astype(np.float32)
    repeated_sigmas = np.repeat(sigmas, support_size).reshape((total_size,)).astype(np.float32)

    normalization_constant = np.sqrt(2 * np.pi)
    normalization_constants = np.repeat(1 / (normalization_constant * np.array(sigmas)), support_size).reshape(
        (total_size,))

    ys = (supports - repeated_means) / repeated_sigmas

    return (normalization_constants * np.exp(-(ys ** 2) / 2)).reshape((size, support_size))


def filter_distant_values_in_distribution(pdfs, sigmas, means, support_discretization_factor, lower_bound,
                                          upper_bound, negligible_distance=5):
    lower_negligible = means - (negligible_distance * sigmas)
    upper_negligible = means + (negligible_distance * sigmas)
    negligible_to = ((np.maximum(np.repeat(lower_bound, len(means)),
                                 lower_negligible)) / support_discretization_factor).astype(int)
    negligible_from = ((np.minimum(np.repeat(upper_bound, len(means)),
                                   upper_negligible)) / support_discretization_factor).astype(int)

    for i, n in enumerate(negligible_to):
        pdfs[i, :n] = 0
    for i, n in enumerate(negligible_from):
        pdfs[i, n:] = 0


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


@dataclasses.dataclass(frozen=True)
class NumericCalculator(Calculator):
    numeric2index: Dict[int, int]
    support: Tuple
    reactive_unit_distribution: np.ndarray[np.ndarray[float]]
    reactive_x_reactive: np.ndarray[np.ndarray[float]]

    def domain(self):
        return self.support

    def dot_product(self, r1: int, r2: int):
        i1 = self.numeric2index[r1]
        i2 = self.numeric2index[r2]
        return self.reactive_x_reactive[i1][i2]

    def dot_product_all(self, rs1: List[int]):
        is1 = [self.numeric2index[r1] for r1 in rs1]
        return self.reactive_x_reactive[:, is1]

    def pdf(self, r: int):
        i = self.numeric2index[r]
        return self.reactive_unit_distribution[i]

    def activation_from_responses(self, response_over_stimuli: List[float]):
        return np.array(response_over_stimuli).astype(bool)

    @staticmethod
    def from_description_with_no_ans(sigma=.03):
        support = tuple(np.arange(-5.5, 105.5, .01))

        stimuli = tuple([int(x) for x in np.arange(1, 101).astype(int)])

        pdfs = calculate_normal_pdfs(support, stimuli, np.repeat(sigma, len(stimuli)))

        rxr = np.dot(pdfs, np.transpose(pdfs))

        numeric2index = {v: index for index, v in enumerate(stimuli)}

        return stimuli, NumericCalculator(numeric2index, support, pdfs, rxr)

    @staticmethod
    def from_description_with_ans():
        support = tuple(np.arange(0, 150., .01))

        stimuli = tuple([int(x) for x in np.arange(1, 101).astype(int)])

        pdfs = calculate_normal_pdfs(support, stimuli, np.array(stimuli) * .1)

        rxr = np.dot(pdfs, np.transpose(pdfs))

        numeric2index = {v: index for index, v in enumerate(stimuli)}

        return stimuli, NumericCalculator(numeric2index, support, pdfs, rxr)

    @staticmethod
    def load_from_file_with_ans():
        return NumericCalculator.load_from_file('../inmemory_calculus_ans/numeric')

    @staticmethod
    def load_from_file_with_no_ans():
        return NumericCalculator.load_from_file('../inmemory_calculus_no_ans/numeric')

    @staticmethod
    def load_from_file(path='../inmemory_calculus/franek/numeric'):
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
        domain = tuple(domain)

        stimuli = tuple([*range(1, len(reactive_unit_distribution) + 1)])

        # VALIDATE loaded data shapes:
        if not len(stimuli) == reactive_unit_distribution.shape[0] == reactive_x_reactive.shape[0] == \
               reactive_x_reactive.shape[1]:
            raise ValueError()
        if not reactive_unit_distribution.shape[1] == len(domain):
            raise ValueError()

        numeric2index = {v: index for index, v in enumerate(stimuli)}
        return stimuli, NumericCalculator(numeric2index, domain, reactive_unit_distribution, reactive_x_reactive)


@dataclasses.dataclass(frozen=True)
class QuotientCalculator(Calculator):
    quotient2index: Dict[Tuple[int, int], int]
    support: Tuple
    reactive_unit_distribution: np.ndarray[np.ndarray[float]]
    reactive_x_reactive: np.ndarray[np.ndarray[float]]

    @staticmethod
    def compute_quotient2index(f: QuotientStimulus):
        return f.numerator, f.denominator

    def domain(self):
        return self.support

    def dot_product(self, r1: QuotientStimulus, r2: QuotientStimulus):
        i1 = self.quotient2index[QuotientCalculator.compute_quotient2index(r1)]
        i2 = self.quotient2index[QuotientCalculator.compute_quotient2index(r2)]
        return self.reactive_x_reactive[i1][i2]

    def dot_product_all(self, rs1: List[QuotientStimulus]):
        is1 = [self.quotient2index[QuotientCalculator.compute_quotient2index(r1)] for r1 in rs1]
        return self.reactive_x_reactive[:, is1]

    def pdf(self, r: QuotientStimulus):
        i = self.quotient2index[QuotientCalculator.compute_quotient2index(r)]
        return self.reactive_unit_distribution[i]

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
    def from_description_with_no_ans(sigma=.006):
        fractions = list(set([Fraction(nom, denom) for denom in range(1, 101) for nom in range(1, denom + 1)]))
        fractions = tuple(sorted(fractions))
        support_lower_bound = 0.
        support_upper_bound = 2.
        support_discretization_factor = .001

        support = tuple(np.arange(support_lower_bound, support_upper_bound, support_discretization_factor))

        stimuli = fractions
        stimuli_floats = np.array(fractions).astype(float)  # means
        sigmas = np.repeat(sigma, len(stimuli_floats))

        pdfs = calculate_normal_pdfs(support, stimuli_floats, sigmas)

        filter_distant_values_in_distribution(pdfs, sigmas, stimuli_floats, support_discretization_factor,
                                              support_lower_bound, support_upper_bound)

        rxr = np.dot(pdfs, np.transpose(pdfs))
        quotient2index = {QuotientCalculator.compute_quotient2index(v): index for index, v in enumerate(stimuli)}

        return stimuli, QuotientCalculator(quotient2index, support, pdfs, rxr)

    @staticmethod
    def from_description_with_ans(sigma_scalar=.1):
        support_lower_bound = 0
        support_upper_bound = 2.3
        support_discretization_factor = .001
        support = tuple(np.arange(support_lower_bound, support_upper_bound, support_discretization_factor))

        fractions = list(set([Fraction(nom, denom) for denom in range(1, 101) for nom in range(1, denom + 1)]))
        fractions = tuple(sorted(fractions))

        stimuli = fractions
        stimuli_floats = np.array(fractions).astype(float)
        sigmas = np.array(stimuli_floats) * sigma_scalar
        pdfs = calculate_normal_pdfs(support, stimuli_floats, sigmas)

        filter_distant_values_in_distribution(pdfs, sigmas, stimuli_floats, support_discretization_factor,
                                              support_lower_bound, support_upper_bound)

        rxr = np.dot(pdfs, np.transpose(pdfs))
        quotient2index = {QuotientCalculator.compute_quotient2index(v): index for index, v in enumerate(stimuli)}

        return stimuli, QuotientCalculator(quotient2index, support, pdfs, rxr)

    @staticmethod
    def load_from_file_with_ans():
        return QuotientCalculator.load_from_file(root_path='../inmemory_calculus/franek',
                                                 pdfs_file_name='quotient_discrete_Ri_sigma_5.h5',
                                                 rxr_file_name='quotient_elements.h5',
                                                 support_file_name='x.h5',
                                                 stimuli_file_name='nklist.h5')

    @staticmethod
    def load_from_file_with_no_ans():
        return QuotientCalculator.load_from_file(root_path='../inmemory_calculus_no_ans/quotient',
                                                 pdfs_file_name='R.h5',
                                                 rxr_file_name='RxR.h5',
                                                 support_file_name='domain.h5',
                                                 stimuli_file_name='nklist.h5')

    @staticmethod
    def load_from_file(root_path, pdfs_file_name, rxr_file_name, support_file_name, stimuli_file_name) -> Tuple[
        Tuple, Calculator]:
        root_path = Path(os.path.abspath(root_path))

        reactive_unit_distribution = read_h5_data(data_path=root_path.joinpath(pdfs_file_name))
        if not isinstance(reactive_unit_distribution, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(reactive_unit_distribution)))

        reactive_x_reactive = read_h5_data(root_path.joinpath(rxr_file_name))
        if not isinstance(reactive_x_reactive, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(reactive_x_reactive)))

        domain = read_h5_data(root_path.joinpath(support_file_name))
        if not isinstance(domain, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(domain)))
        domain = tuple(domain)

        # reduced fractions n/k where n < k and k <= 100; reduced def: n, k are relatively prime integers
        stimuli = read_h5_data(root_path.joinpath(stimuli_file_name))
        # invoke int(.) for serialization reasons (int64 is not json serializable)
        stimuli = tuple(Fraction(int(nom), int(denom)) for nom, denom in stimuli)

        # VALIDATE loaded data shapes:
        if not len(stimuli) == reactive_unit_distribution.shape[0] == reactive_x_reactive.shape[0] == \
               reactive_x_reactive.shape[1]:
            raise ValueError()
        if not reactive_unit_distribution.shape[1] == len(domain):
            raise ValueError()

        quotient2index = {QuotientCalculator.compute_quotient2index(v): index for index, v in enumerate(stimuli)}
        return stimuli, QuotientCalculator(quotient2index, domain, reactive_unit_distribution, reactive_x_reactive)


def load_stimuli_and_calculator(stimuli_type, with_ans=True):
    assert stimuli_type in {'quotient', 'numeric'}
    if stimuli_type == 'quotient' and with_ans:
        return QuotientCalculator.from_description_with_ans()
    if stimuli_type == 'quotient' and not with_ans:
        return QuotientCalculator.from_description_with_no_ans()
    if stimuli_type == 'numeric' and with_ans:
        return NumericCalculator.from_description_with_ans()
    if stimuli_type == 'numeric' and not with_ans:
        return NumericCalculator.from_description_with_no_ans()


if __name__ == '__main__':
    # s = time.time()
    # a = QuotientCalculator.from_description_with_ans()
    # print(time.time()-s)
    # s = time.time()
    # a = QuotientCalculator.from_description_with_ans()
    # print(time.time()-s)
    # root_path = Path(os.path.abspath('../inmemory_calculus/franek'))
    # elements = read_h5_data(data_path=root_path.joinpath('elements.h5'))
    # numeric_elements = read_h5_data(data_path=root_path.joinpath('numeric_elements.h5'))
    # print(numeric_elements)
    QuotientCalculator.from_description_with_ans()
