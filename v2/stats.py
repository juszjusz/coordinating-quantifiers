from typing import List
import numpy as np
from numpy import mean
from numpy import std
from scipy import stats
from math import sqrt


def confidence_interval(sample, level=0.95, method='t'):
    mean_val = mean(sample)
    n = len(sample)
    stdev = std(sample)
    if method == 't':
        test_stat = stats.t.ppf((level + 1) / 2, n)
    elif method == 'z':
        test_stat = stats.norm.ppf((level + 1) / 2)
    lower_bound = mean_val - test_stat * stdev / sqrt(n)
    upper_bound = mean_val + test_stat * stdev / sqrt(n)

    return lower_bound, upper_bound


def new_confidence_intervals(sample_series: np.ndarray, level=0.95, method='t'):
    shape = sample_series.shape
    assert len(shape) == 2, 'function computes confidence intervals for list of sequences'
    n, sequence_length = shape

    means = np.mean(sample_series, axis=0)
    stdevs = np.std(sample_series, axis=0)

    if method == 't':
        test_stat = stats.t.ppf((level + 1) / 2, n)
    elif method == 'z':
        test_stat = stats.norm.ppf((level + 1) / 2)
    else:
        raise NotImplementedError(f'{method} is an unknown method for computing CI, implemented for "t" and "z"')

    margins = (test_stat * stdevs) / sqrt(n)

    lower_bounds = means - margins
    upper_bounds = means + margins

    return lower_bounds, upper_bounds


def means(sample_series):
    return [mean(sample_series[t]) for t in range(len(sample_series))]


def confidence_intervals(sample_series):
    cis = [confidence_interval(sample_series[t]) for t in range(len(sample_series))]
    cis_l = [i[0] for i in cis]
    cis_u = [i[1] for i in cis]
    return cis_l, cis_u


if __name__ == '__main__':
    a = np.array([[0, 4, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 9],
                  [0, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 9, 11],
                  [0, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 8, 8],
                  [0, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 8, 10],
                  [0, 4, 6, 6, 6, 5, 5, 5, 5, 5, 6, 7, 10],
                  [0, 4, 4, 5, 5, 5, 5, 6, 5, 6, 6, 6, 7],
                  [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7],
                  [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 8],
                  [0, 5, 5, 4, 4, 3, 4, 4, 6, 7, 8, 10, 10],
                  [0, 6, 6, 5, 5, 5, 5, 5, 6, 7, 7, 8, 9],
                  [0, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8],
                  [0, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 8],
                  [0, 5, 5, 7, 6, 7, 7, 7, 7, 7, 7, 9, 10],
                  [0, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 9],
                  [0, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 9],
                  [0, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7, 10],
                  [0, 4, 4, 4, 5, 5, 5, 5, 5, 7, 7, 8, 8],
                  [0, 5, 4, 5, 5, 5, 5, 7, 7, 7, 7, 9, 10],
                  [0, 5, 4, 4, 5, 5, 5, 5, 5, 5, 6, 7, 10],
                  [0, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6]])
    # x = new_confidence_intervals(a)
    x = new_confidence_intervals(a)
    print(x)
