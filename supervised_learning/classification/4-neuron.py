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

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression.

        Args:
            Y (np.ndarray): Correct labels for the input data,
                shape (1, m).
            A (np.ndarray): Activated output of the neuron for each
                example, shape (1, m).

        Returns:
            float: The cost of the model.
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) *
                             np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions.

        Args:
            X (np.ndarray): Input data with shape (nx, m).
            Y (np.ndarray): Correct labels for the input data,
                shape (1, m).

        Returns:
            tuple: The neuron's prediction (shape (1, m)) and
                the cost of the network.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
