#!/usr/bin/env python3
"""Moving average function."""


def moving_average(data, beta):
    """Calculate the weighted moving average of a data set.

    Args:
        data (list): List of data to calculate the moving average of.
        beta (float): Weight used for the moving average.

    Returns:
        list: List containing the moving averages of data.
    """
    v = 0
    moving_averages = []

    for t, value in enumerate(data, 1):
        v = beta * v + (1 - beta) * value
        # Bias correction
        v_corrected = v / (1 - beta ** t)
        moving_averages.append(v_corrected)

    return moving_averages
