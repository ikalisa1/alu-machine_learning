#!/usr/bin/env python3
"""Module for RMSProp optimization algorithm."""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update a variable using the RMSProp optimization algorithm.

    Args:
        alpha: The learning rate
        beta2: The RMSProp weight (decay rate)
        epsilon: A small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        s: The previous second moment of var

    Returns:
        The updated variable and the new moment, respectively
    """
    s_new = beta2 * s + (1 - beta2) * np.square(grad)
    var_updated = var - alpha * grad / (np.sqrt(s_new) + epsilon)
    return var_updated, s_new
