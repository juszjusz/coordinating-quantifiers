import numpy as np
from scipy.integrate import quad


def interpolate(x, y):
    return lambda val: np.interp(val, x, y)


def integrate(fun, x_left, x_right):
    return quad(fun, x_left, x_right, limit=5)[0]
