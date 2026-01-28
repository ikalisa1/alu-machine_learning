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
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

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

    def forward_prop(self, X):
        """Calculate forward propagation of the neuron.

        Args:
            X (np.ndarray): Input data with shape (nx, m).
                nx is the number of input features.
                m is the number of examples.

        Returns:
            np.ndarray: The private attribute __A (activation output).
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
