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


def means(sample_series):
    return [mean(sample_series[t]) for t in range(len(sample_series))]


def confidence_intervals(sample_series):
    cis = [confidence_interval(sample_series[t]) for t in range(len(sample_series))]
    cis_l = [i[0] for i in cis]
    cis_u = [i[1] for i in cis]
    return cis_l, cis_u
