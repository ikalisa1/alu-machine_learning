#!/usr/bin/env python3
"""
add_arrays = __import__('4-line_up').add_arrays

arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]
print(add_arrays(arr1, arr2))
print(arr1)
print(arr2)
print(add_arrays(arr1, [1, 2, 3]))
"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise.

    Args:
        arr1: A list of integers or floats.
        arr2: A list of integers or floats.

    Returns:
        A new list with the element-wise sum of arr1 and arr2.
        If arr1 and arr2 are not the same shape, returns None.
    """
    if len(arr1) != len(arr2):
        return None  # Return None if arrays are not the same length

    return [arr1[i] + arr2[i] for i in range(len(arr1))]
