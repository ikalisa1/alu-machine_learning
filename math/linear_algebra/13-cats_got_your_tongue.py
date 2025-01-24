#!/usr/bin/env python3
"""This module concatenates two matrices along a specific axis."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis.

    Args:
        mat1: A numpy array.
        mat2: A numpy array.
        axis: The axis along which to concatenate. Defaults to 0 (rows).

    Returns:
        A new numpy array that results from concatenating mat1 and mat2.
    """
    return np.concatenate((mat1, mat2), axis=axis)
