#!/usr/bin/env python3
"""Module for creating confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes) containing
                the correct labels for each data point
        logits: one-hot numpy.ndarray of shape (m, classes) containing
                the predicted labels

    Returns:
        confusion numpy.ndarray of shape (classes, classes) with row indices
        representing the correct labels and column indices representing
        the predicted labels
    """
    # Convert one-hot encoded arrays to class indices
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)
    
    # Get number of classes
    classes = labels.shape[1]
    
    # Initialize confusion matrix
    confusion = np.zeros((classes, classes))
    
    # Populate confusion matrix
    for true, pred in zip(true_labels, pred_labels):
        confusion[true, pred] += 1
    
    return confusion
