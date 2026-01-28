#!/usr/bin/env python3
"""Module for batch normalization."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalize an unactivated output of a neural network using batch
    normalization.

    Args:
        Z: numpy.ndarray of shape (m, n) that should be normalized
        gamma: numpy.ndarray of shape (1, n) containing the scales
        beta: numpy.ndarray of shape (1, n) containing the offsets
        epsilon: A small number used to avoid division by zero

    Returns:
        The normalized Z matrix
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_output = gamma * Z_norm + beta
    return Z_output
