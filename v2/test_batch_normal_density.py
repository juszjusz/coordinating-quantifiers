# def calculate_normal_pdfs(support: Support, means: Stimuli, sigmas: list[float]):
#     assert len(means) == len(sigmas), "expects means & sigmas to be of equal sizes"
#     support_size = len(support)
#     size = len(means)
#     total_size = support_size * size
#
#     supports = np.tile(support, size).reshape((total_size,))
#     repeated_means = np.repeat(means, support_size).reshape((total_size,))
#     repeated_sigmas = np.repeat(sigmas, support_size).reshape((total_size,))
#
#     normalization_constant = np.sqrt(2 * np.pi)
#     normalization_constants = np.repeat(1 / (normalization_constant * np.array(sigmas)), support_size).reshape(
#         (total_size,))
#
#     ys = (supports - repeated_means) / repeated_sigmas
#
#     return (normalization_constants * np.exp(-(ys ** 2) / 2)).reshape((size, support_size))
import time
import unittest
import numpy as np
from scipy.stats import norm

from calculator import Support, calculate_normal_pdfs, Stimuli


class TestBatchNormalDensity(unittest.TestCase):
    def test(self):
        support_lower_bound = 0
        support_upper_bound = 2.3
        support_discretization_factor = .001

        support: Support = tuple(np.arange(support_lower_bound, support_upper_bound, support_discretization_factor))

        size = 5000
        means: Stimuli = tuple(np.arange(0, size))
        sigmas = np.repeat([1], size)
        start_time = time.time()
        pdfs = calculate_normal_pdfs(support, means, sigmas)
        end_time = time.time()
        print("batch normal calc time took", end_time - start_time)
        start_time = time.time()
        pdfs_expected = np.array([norm.pdf(support, mu, sigma) for mu, sigma in zip(means, sigmas)])
        end_time = time.time()
        print("iterable normal calc time took", end_time - start_time)

        for actual, expected in zip(pdfs, pdfs_expected):
            max_diff = np.max(abs(actual - expected))
            self.assertTrue(max_diff < 0.0001)
