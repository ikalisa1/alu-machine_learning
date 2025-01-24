#!/usr/bin/env python3
"""
cat_arrays = __import__('6-howdy_partner').cat_arrays

arr1 = [1, 2, 3, 4, 5]
arr2 = [6, 7, 8]
print(cat_arrays(arr1, arr2))
print(arr1)
print(arr2)
"""


def cat_arrays(arr1, arr2):
    """Concatenates two arrays.

    Args:
        arr1: A list of integers or floats.
        arr2: A list of integers or floats.

    Returns:
        A new list that is the concatenation of arr1 and arr2.
    """
    return arr1 + arr2
