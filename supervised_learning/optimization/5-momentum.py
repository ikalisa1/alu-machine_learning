#!/usr/bin/env python3
"""Momentum optimization function."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using gradient descent with momentum optimization.

    Args:
        alpha (float): Learning rate.
        beta1 (float): Momentum weight.
        var (np.ndarray): Variable to be updated.
        grad (np.ndarray): Gradient of var.
        v (np.ndarray): Previous first moment of var.

    Returns:
        tuple: The updated variable and the new moment.
    """
    v_new = beta1 * v + (1 - beta1) * grad
    var_updated = var - alpha * v_new
    return var_updated, v_new
