from typing import Any

import numpy as np


def base_julia_iteration(
        z: np.ndarray[int, np.complex128],
        to_compute: np.ndarray[Any, np.dtype[bool]],
        c: np.complex128
):
    """
    Applies the iteration
    :param z: complex number grid
    :param to_compute: boolean mask indicating which elements to compute
    :param c: complex constant for the iteration
    :return: the updated grid of complex numbers
    """

    z[to_compute] = (z[to_compute] ** 2)
    z[to_compute] += c
    return z


def base_julia_escape_time(
        z: np.ndarray[int, np.complex128],
        to_compute: np.ndarray[Any, np.dtype[bool]]
):
    """
    With a Julia set, we deem a point to be divergent if its module
    is bigger than 2. In the general case we compute the escape time as the maximum value
    between 2*|a_n| and 2(sum(absolute value of the coefficients of the polynomial)/|a_n|).

    In the simple case just check whether a point is inside the radius two disk, using the usual
    absolute value
    :param z: complex number grid
    :param to_compute: boolean mask indicating which elements to compute
    :return: a matrix which indicates which point will be divergent.
    """

    diverging = np.zeros((len(z), len(z[0])), dtype=bool)
    diverging[to_compute] = np.abs(z[to_compute]) > 2
    return diverging
