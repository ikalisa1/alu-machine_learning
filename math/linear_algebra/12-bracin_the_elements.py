#!/usr/bin/env python3
"""
Module that performs element-wise operations.
"""


def np_elementwise(mat1, mat2):
    """Performs element-wise opreations.

    Args:
        mat1: A NumPy array.
        mat2: A NumPy array or scalar.

    Returns:
        A tuple with element-wise sum, difference, product, and quotient.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
