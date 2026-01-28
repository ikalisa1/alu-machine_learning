#!/usr/bin/env python3
"""Neuron class for binary classification."""
import numpy as np


class Neuron:
    """Define a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize a neuron.

        Args:
            nx (int): Number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return the weights of the neuron.

        Returns:
            np.ndarray: The weights of the neuron.
        """
        return self.__W

    @property
    def b(self):
        """Return the bias of the neuron.

        Returns:
            int: The bias of the neuron.
        """
        return self.__b

    @property
    def A(self):
        """Return the activation output of the neuron.

        Returns:
            int or float: The activation output of the neuron.
        """
        return self.__A

    @A.setter
    def A(self, value):
        """Set the activation output of the neuron.

        Args:
            value (int or float): The value to set for activation.
        """
        self.__A = value
