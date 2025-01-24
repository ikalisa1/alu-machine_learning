#!/usr/bin/env python3
"""
Module for calculating the shape of a NumPy array.

np_shape function calculates the shape of a NumPy array.
"""


def np_shape(matrix):
    """Calculates the shape of a numpy.ndarray.

    Args:
        matrix: A numpy array whose shape needs to be calculated.

    Returns:
        A tuple of integers representing the shape of the numpy array.
    """
    return tuple(matrix.shape)
