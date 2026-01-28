#!/usr/bin/env python3
"""Normalization constants function."""
import numpy as np


def normalization_constants(X):
    """Calculate the normalization (standardization) constants of a matrix.

    Args:
        X (np.ndarray): Matrix of shape (m, nx) to normalize.
            m is the number of data points.
            nx is the number of features.

    Returns:
        tuple: The mean and standard deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
