#!/usr/bin/env python3
"""
add_matrices2D = __import__('5-across_the_planes').add_matrices2D

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
print(add_matrices2D(mat1, mat2))
print(mat1)
print(mat2)
print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise.

    Args:
        mat1: A 2D list (matrix) of integers or floats.
        mat2: A 2D list (matrix) of integers or floats.

    Returns:
        A new 2D list with the element-wise sum of mat1 and mat2.
        If mat1 and mat2 are not the same shape, returns None.
    """
    if len(mat1) != len(mat2):
        return None

    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None

    return [
        [row1[i] + row2[i] for i in range(len(row1))]
        for row1, row2 in zip(mat1, mat2)
    ]
