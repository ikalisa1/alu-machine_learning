#!/usr/bin/env python3
"""Module for calculating precision"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes) where
                   row indices represent the correct labels and column indices
                   represent the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the precision of
        each class
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
