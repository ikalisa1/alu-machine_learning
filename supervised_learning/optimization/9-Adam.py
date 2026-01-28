#!/usr/bin/env python3
"""Module for Adam optimization algorithm."""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Update a variable using the Adam optimization algorithm.

    Args:
        alpha: The learning rate
        beta1: The weight used for the first moment
        beta2: The weight used for the second moment
        epsilon: A small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: The previous first moment of var
        s: The previous second moment of var
        t: The time step used for bias correction

    Returns:
        The updated variable, the new first moment, and the new second moment
    """
    # Update first moment (momentum)
    v_new = beta1 * v + (1 - beta1) * grad

    # Update second moment (RMSProp)
    s_new = beta2 * s + (1 - beta2) * np.square(grad)

    # Bias correction
    v_corrected = v_new / (1 - np.power(beta1, t))
    s_corrected = s_new / (1 - np.power(beta2, t))

    # Update variable
    var_updated = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var_updated, v_new, s_new
