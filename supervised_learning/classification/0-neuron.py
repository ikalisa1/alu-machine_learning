#!/usr/bin/env python3
"""
Neuron class for binary classification
"""

import numpy as np


class Neuron:
    """
    Define a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize the neuron.

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

        # Private attributes
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for the weights.

        Returns:
            np.ndarray: The weights of the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the bias.

        Returns:
            int: The bias of the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the activation output.

        Returns:
            int or float: The activation output of the neuron.
        """
        return self.__A

    @A.setter
    def A(self, value):
        """
        Setter for the activation output.

        Args:
            value (int or float): The value to set for the activation output.
        """
        self.__A = value
