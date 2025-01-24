#!/usr/bin/env python3
"""
cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
print(mat4)
print(mat5)
mat1[0] = [9, 10]
mat1[1].append(5)
print(mat1)
print(mat4)
print(mat5)
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along a specific axis.

    Args:
        mat1: A 2D list (matrix) of integers or floats.
        mat2: A 2D list (matrix) of integers or floats.
        axis: The axis along which to concatenate the matrices.
        axis=0 means row-wise, axis=1 means column-wise.

    Returns:
       If the matrices cannot be concatenated, returns None.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None
