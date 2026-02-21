#!/usr/bin/env python3
"""Module for calculating specificity"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes) where
                   row indices represent the correct labels and column indices
                   represent the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the specificity of
        each class
    """
    classes = confusion.shape[0]
    specificity_values = np.zeros(classes)
    
    for i in range(classes):
        true_positive = confusion[i, i]
        false_positive = np.sum(confusion[:, i]) - true_positive
        false_negative = np.sum(confusion[i, :]) - true_positive
        true_negative = np.sum(confusion) - true_positive - false_positive - false_negative
        
        specificity_values[i] = true_negative / (true_negative + false_positive)
    
    return specificity_values
