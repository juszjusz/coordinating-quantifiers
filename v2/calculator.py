import dataclasses
import os
from fractions import Fraction
from functools import singledispatch, lru_cache
from pathlib import Path
from typing import List, Tuple, Union, Callable, Any, Dict

import h5py
import numpy as np

NumericStimulus = int
QuotientStimulus = Fraction

Stimulus = Union[NumericStimulus, QuotientStimulus]
Stimuli = tuple[Stimulus, ...]
NumericStimulusContext = tuple[NumericStimulus, NumericStimulus]
QuotientStimulusContext = tuple[QuotientStimulus, QuotientStimulus]
StimulusContext = Union[NumericStimulusContext, QuotientStimulusContext]
Support = tuple[float, ...]


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


def calculate_normal_pdfs(support: Support, means: Stimuli, sigmas: list[float]):
    assert len(means) == len(sigmas), "expects means & sigmas to be of equal sizes"
    support_size = len(support)
    size = len(means)
    total_size = support_size * size

    supports = np.tile(support, size).reshape((total_size,))
    repeated_means = np.repeat(means, support_size).reshape((total_size,))
    repeated_sigmas = np.repeat(sigmas, support_size).reshape((total_size,))

    normalization_constant = np.sqrt(2 * np.pi)
    normalization_constants = np.repeat(1 / (normalization_constant * np.array(sigmas)), support_size).reshape(
        (total_size,))

    ys = (supports - repeated_means) / repeated_sigmas

    return (normalization_constants * np.exp(-(ys ** 2) / 2)).reshape((size, support_size))


def filter_distant_values_in_distribution(pdfs, sigmas, means, support_discretization_factor, lower_bound,
                                          upper_bound, negligible_distance_in_sigma=5):
    lower_negligible = means - (negligible_distance_in_sigma * sigmas) - lower_bound
    upper_negligible = means + (negligible_distance_in_sigma * sigmas) - lower_bound

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


@dataclasses.dataclass
class StimuliDensity:
    def __init__(self, index_mapping: dict[Stimulus, int], support: Support,
                 reactive_unit_distribution: np.ndarray[np.ndarray[float]]):
        self.index_mapping = index_mapping
        self._support = support
        self.reactive_unit_distribution = reactive_unit_distribution

    def pdf(self, stimulus: Stimulus) -> np.ndarray[float]:
        i = self.index_mapping[stimulus]
        return self.reactive_unit_distribution[i]

    def support(self) -> Support:
        return self._support


class Calculator:

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
    numeric2index: dict[int, int]
    reactive_x_reactive: np.ndarray[np.ndarray[float]]

    def dot_product(self, r1: int, r2: int):
        i1 = self.numeric2index[r1]
        i2 = self.numeric2index[r2]
        return self.reactive_x_reactive[i1][i2]

    def dot_product_all(self, rs1: list[int]):
        is1 = [self.numeric2index[r1] for r1 in rs1]
        return self.reactive_x_reactive[:, is1]

    def activation_from_responses(self, response_over_stimuli: list[float]):
        return np.array(response_over_stimuli).astype(bool)

    @staticmethod
    def from_description_with_no_ans(sigma=1 / 3, negligible_distance_in_sigma=5, stimuli_range=20):
        support_lower_bound = -5
        support_upper_bound = 105
        support_discretization_factor = .01

        support: Support = tuple(np.arange(support_lower_bound, support_upper_bound, support_discretization_factor))

        stimuli: Stimuli = tuple([int(x) for x in np.arange(1, stimuli_range + 1).astype(int)])

        sigmas = np.repeat(sigma, len(stimuli))

        pdfs = calculate_normal_pdfs(support, stimuli, sigmas)

        rxr = np.dot(pdfs, np.transpose(pdfs))
        filter_distant_values_in_distribution(pdfs, sigmas, stimuli, support_discretization_factor,
                                              support_lower_bound, support_upper_bound, negligible_distance_in_sigma)

        numeric2index = {v: index for index, v in enumerate(stimuli)}

        return stimuli, StimuliDensity(numeric2index, support, pdfs), NumericCalculator(numeric2index, rxr)

    @staticmethod
    def from_description_with_ans(sigma_scalar=.1, negligible_distance_in_sigma=5, stimuli_range=20):
        support_lower_bound = 0
        support_upper_bound = 150
        support_discretization_factor = .01
        support = tuple(np.arange(support_lower_bound, support_upper_bound, support_discretization_factor))

        stimuli = tuple([int(x) for x in np.arange(1, stimuli_range + 1).astype(int)])

        sigmas = np.array(stimuli) * sigma_scalar
        pdfs = calculate_normal_pdfs(support, stimuli, sigmas)

        rxr = np.dot(pdfs, np.transpose(pdfs))
        filter_distant_values_in_distribution(pdfs, sigmas, stimuli, support_discretization_factor,
                                              support_lower_bound, support_upper_bound, negligible_distance_in_sigma)

        numeric2index = {v: index for index, v in enumerate(stimuli)}

        return stimuli, StimuliDensity(numeric2index, support, pdfs), NumericCalculator(numeric2index, rxr)

    @staticmethod
    def load_from_file_with_ans():
        return NumericCalculator.load_from_file('./inmemory_calculus_ans/numeric')

    @staticmethod
    def load_from_file_with_no_ans():
        return NumericCalculator.load_from_file('./inmemory_calculus_no_ans/numeric')

    @staticmethod
    def load_from_file(path='../inmemory_calculus/franek/numeric'):
        root_path = Path(os.path.abspath(path))

        reactive_unit_distribution = read_h5_data(data_path=root_path.joinpath('R.h5'))
        if not isinstance(reactive_unit_distribution, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(reactive_unit_distribution)))

        reactive_x_reactive = read_h5_data(root_path.joinpath('RxR.h5'))
        if not isinstance(reactive_x_reactive, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(reactive_x_reactive)))

        support = read_h5_data(root_path.joinpath('domain.h5'))
        if not isinstance(support, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(support)))
        support: Support = tuple(support)

        stimuli: Stimuli = tuple([*range(1, len(reactive_unit_distribution) + 1)])

        # VALIDATE loaded data shapes:
        if not len(stimuli) == reactive_unit_distribution.shape[0] == reactive_x_reactive.shape[0] == \
               reactive_x_reactive.shape[1]:
            raise ValueError()
        if not reactive_unit_distribution.shape[1] == len(support):
            raise ValueError()

        numeric2index = {v: index for index, v in enumerate(stimuli)}
        return stimuli, StimuliDensity(numeric2index, support, reactive_unit_distribution), NumericCalculator(
            numeric2index, reactive_x_reactive)


@dataclasses.dataclass(frozen=True)
class QuotientCalculator(Calculator):
    quotient2index: dict[QuotientStimulus, int]
    reactive_x_reactive: np.ndarray[np.ndarray[float]]

    @staticmethod
    def compute_quotient2index(f: QuotientStimulus):
        return f.numerator, f.denominator

    def dot_product(self, r1: QuotientStimulus, r2: QuotientStimulus):
        i1 = self.quotient2index[r1]
        i2 = self.quotient2index[r2]
        return self.reactive_x_reactive[i1][i2]

    def dot_product_all(self, rs1: List[QuotientStimulus]):
        is1 = [self.quotient2index[r1] for r1 in rs1]
        return self.reactive_x_reactive[:, is1]

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
    def calculate_and_sort_normalized_fractions() -> tuple[Fraction, ...]:
        fractions = set([Fraction(nom, denom) for denom in range(1, 101) for nom in range(1, denom + 1)])
        return tuple(sorted(fractions))

    @staticmethod
    @lru_cache
    def from_description_with_no_ans(sigma=1 / 3, negligible_distance_in_sigma=4):
        normalized_and_sorted_fractions = QuotientCalculator.calculate_and_sort_normalized_fractions()
        sigmas = np.repeat(sigma, len(normalized_and_sorted_fractions))
        fractions, sigmas = QuotientCalculator.calculate_quotient_dist_params_for_quotient_without_ans(
            normalized_and_sorted_fractions, sigmas)

        support_lower_bound = 0.
        support_upper_bound = 2.
        support_discretization_factor = .001

        support = tuple(np.arange(support_lower_bound, support_upper_bound, support_discretization_factor))

        stimuli = fractions
        stimuli_floats = np.array(fractions).astype(float)  # means

        pdfs = calculate_normal_pdfs(support, stimuli_floats, sigmas)

        rxr = np.dot(pdfs, np.transpose(pdfs))
        filter_distant_values_in_distribution(pdfs, sigmas, stimuli_floats, support_discretization_factor,
                                              support_lower_bound, support_upper_bound, negligible_distance_in_sigma)

        quotient2index = {stimuli: index for index, stimuli in enumerate(stimuli)}

        return stimuli, StimuliDensity(quotient2index, support, pdfs), QuotientCalculator(quotient2index, rxr)

    @staticmethod
    @lru_cache
    def from_description_with_ans(sigma_scalar=.1, negligible_distance_in_sigma=4):
        normalized_and_sorted_fractions: tuple[
            Fraction, ...] = QuotientCalculator.calculate_and_sort_normalized_fractions()
        number_sigmas = [sigma_scalar * mean for mean in normalized_and_sorted_fractions]
        # sigmas = np.array([(f.numerator * sigma_scalar) / (f.denominator * sigma_scalar) for f in
        #           normalized_and_sorted_fractions])

        # fractions, sigmas = QuotientCalculator.calculate_quotient_dist_params_for_quotient_with_ans1(
        #     normalized_and_sorted_fractions,
        #     sigmas)

        fractions, sigmas = QuotientCalculator.calculate_quotient_dist_params_for_quotient_with_ans(
            normalized_and_sorted_fractions,
            number_sigmas)

        support_lower_bound = 0
        support_upper_bound = 2.3
        support_discretization_factor = .001
        support = tuple(np.arange(support_lower_bound, support_upper_bound, support_discretization_factor))

        stimuli = fractions
        stimuli_floats = np.array(fractions).astype(float)

        pdfs = calculate_normal_pdfs(support, stimuli_floats, sigmas)

        rxr = np.dot(pdfs, np.transpose(pdfs))
        # filter_distant_values_in_distribution(pdfs, sigmas, stimuli_floats, support_discretization_factor,
        #                                       support_lower_bound, support_upper_bound, negligible_distance_in_sigma)

        quotient2index = {fraction: index for index, fraction in enumerate(stimuli)}

        return stimuli, StimuliDensity(quotient2index, support, pdfs), QuotientCalculator(quotient2index, rxr)

    @staticmethod
    def calculate_quotient_dist_params_for_quotient_without_ans(normalized_and_sorted_fractions,
                                                                sigmas: List[float],
                                                                sample_size=10_000):
        # Here randomness is used for density approximation, hence it is fixed.
        random_state = np.random.RandomState(seed=1)

        nominators = np.array([(f.numerator * (100 / f.denominator)) for f in normalized_and_sorted_fractions])
        denominators = np.repeat(100, len(normalized_and_sorted_fractions))
        numerator_samples = random_state.normal(nominators[:, np.newaxis], sigmas[:, np.newaxis],
                                                (len(nominators), sample_size))
        denominator_samples = random_state.normal(denominators[:, np.newaxis], sigmas[:, np.newaxis],
                                                  (len(denominators), sample_size))

        distribution_means = []
        distribution_sigmas = []
        for f, nk, denominator in zip(normalized_and_sorted_fractions, numerator_samples, denominator_samples):
            samples = np.divide(nk, denominator)
            distribution_means.append(f)
            distribution_sigmas.append(np.std(samples, ddof=1))

        return np.array(distribution_means), np.array(distribution_sigmas)

    @staticmethod
    def calculate_quotient_dist_params_for_quotient_with_ans1(normalized_and_sorted_fractions,
                                                              sigmas: List[float],
                                                              sample_size=10_000):
        # Here randomness is used for density approximation, hence it is fixed.
        random_state = np.random.RandomState(seed=1)

        nominators = np.array([(f.numerator * (10 / f.denominator)) for f in normalized_and_sorted_fractions])
        denominators = np.repeat(10, len(normalized_and_sorted_fractions))
        numerator_samples = random_state.normal(nominators[:, np.newaxis], sigmas[:, np.newaxis],
                                                (len(nominators), sample_size))
        denominator_samples = random_state.normal(denominators[:, np.newaxis], sigmas[:, np.newaxis],
                                                  (len(denominators), sample_size))

        distribution_means = []
        distribution_sigmas = []
        for f, nk, denominator in zip(normalized_and_sorted_fractions, numerator_samples, denominator_samples):
            samples = np.divide(nk, denominator)
            distribution_means.append(f)
            distribution_sigmas.append(np.std(samples, ddof=1))

        return np.array(distribution_means), np.array(distribution_sigmas)

    @staticmethod
    def calculate_quotient_dist_params_for_quotient_with_ans(number_means: tuple[Fraction, ...],
                                                             number_sigmas: list[float],
                                                             sample_size=10_000):
        # Here randomness is used for density approximation, hence it is fixed.
        random_state = np.random.RandomState(seed=1)
        # fractions = set([Fraction(nom, denom) for denom in number_means for nom in range(1, denom + 1)])
        # fractions = tuple(sorted(fractions))

        number_means_as_float = np.array(number_means).astype(float)
        number_sigmas = np.array(number_sigmas)
        numerator_samples = random_state.normal(number_means_as_float[:, np.newaxis], number_sigmas[:, np.newaxis],
                                                (len(number_means), sample_size))
        denominator_samples = random_state.normal(number_means_as_float[:, np.newaxis], number_sigmas[:, np.newaxis],
                                                  (len(number_means), sample_size))
        distribution_means = []
        distribution_sigmas = []
        for f in number_means:
            numerator_idx = f.numerator - 1
            denominator_idx = f.denominator - 1
            n = numerator_samples[numerator_idx]
            k = denominator_samples[denominator_idx]
            samples = np.divide(n, k)
            distribution_means.append(f)
            distribution_sigmas.append(np.std(samples, ddof=1))

        return np.array(distribution_means), np.array(distribution_sigmas)

    @staticmethod
    def load_from_file_with_ans() -> tuple[Stimuli, StimuliDensity, Calculator]:
        return QuotientCalculator.load_from_file(root_path='./inmemory_calculus/quotient',
                                                 # pdfs_file_name='quotient_discrete_Ri_sigma_5.h5',
                                                 pdfs_file_name='R.h5',
                                                 rxr_file_name='RxR.h5',
                                                 support_file_name='domain.h5',
                                                 stimuli_file_name='nklist.h5')

    @staticmethod
    def load_from_file_with_no_ans() -> tuple[Stimuli, StimuliDensity, Calculator]:
        return QuotientCalculator.load_from_file(root_path='./inmemory_calculus_no_ans/quotient',
                                                 pdfs_file_name='R.h5',
                                                 rxr_file_name='RxR.h5',
                                                 support_file_name='domain.h5',
                                                 stimuli_file_name='nklist.h5')

    @staticmethod
    def load_from_file(root_path, pdfs_file_name, rxr_file_name, support_file_name, stimuli_file_name) -> tuple[
        Stimuli, StimuliDensity, Calculator]:
        root_path = Path(os.path.abspath(root_path))

        reactive_unit_distribution = read_h5_data(data_path=root_path.joinpath(pdfs_file_name))
        if not isinstance(reactive_unit_distribution, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(reactive_unit_distribution)))

        reactive_x_reactive = read_h5_data(root_path.joinpath(rxr_file_name))
        if not isinstance(reactive_x_reactive, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(reactive_x_reactive)))

        support = read_h5_data(root_path.joinpath(support_file_name))
        if not isinstance(support, np.ndarray):
            raise ValueError('Expected ? to be numpy array, found {} type'.format(type(support)))
        support: Support = tuple(support)

        # reduced fractions n/k where n < k and k <= 100; reduced def: n, k are relatively prime integers
        stimuli = read_h5_data(root_path.joinpath(stimuli_file_name))
        # invoke int(.) for serialization reasons (int64 is not json serializable)
        stimuli: Stimuli = tuple(Fraction(int(nom), int(denom)) for nom, denom in stimuli)

        # VALIDATE loaded data shapes:
        if not len(stimuli) == reactive_unit_distribution.shape[0] == reactive_x_reactive.shape[0] == \
               reactive_x_reactive.shape[1]:
            raise ValueError()
        if not reactive_unit_distribution.shape[1] == len(support):
            raise ValueError()

        # quotient2index = {QuotientCalculator.compute_quotient2index(v): index for index, v in enumerate(stimuli)}
        quotient2index = {v: index for index, v in enumerate(stimuli)}
        return stimuli, StimuliDensity(quotient2index, support, reactive_unit_distribution), QuotientCalculator(
            quotient2index, reactive_x_reactive)


def load_stimuli_and_calculator(stimuli_type, with_ans=True) -> [Tuple, StimuliDensity, Calculator]:
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
    # QuotientCalculator.from_description_with_ans()
    support, density, calculator = QuotientCalculator.from_description_with_ans()
    # np.dot(pdfs[1352, :], np.transpose(pdfs))
    # calculator.dot_product_all([Fraction(43, 97)])
    # QuotientCalculator.from_description_with_ans()
