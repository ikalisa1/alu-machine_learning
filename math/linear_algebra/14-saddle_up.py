#!/usr/bin/env python3
"""This module performs matrix multiplication of two numpy arrays.
"""
import numpy as np


def np_matmul(mat1, mat2):
    """Performs matrix multiplication of two numpy arrays.

    Args:
        mat1: A NumPy array.
        mat2: A NumPy array.

    Returns:
        Result from the matrix multiplication of mat1 and mat2.
    """
    return np.matmul(mat1, mat2)
