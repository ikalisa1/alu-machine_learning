#!/usr/bin/env python3
"""Normalization function."""
import numpy as np


def normalize(X, m, s):
    """Normalize (standardize) a matrix.

    Args:
        X (np.ndarray): Matrix of shape (d, nx) to normalize.
            d is the number of data points.
            nx is the number of features.
        m (np.ndarray): Mean of all features of X, shape (nx,).
        s (np.ndarray): Standard deviation of all features of X,
            shape (nx,).

    Returns:
        np.ndarray: The normalized X matrix.
    """
    return (X - m) / s
