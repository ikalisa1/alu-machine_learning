#!usr/bin/env python3
"""Neuron class for binary classification"""
import numpy as np
class Neuron:
    def_init_(self, nx):
   

    if not isinstance(nx, int):
        raise TypeError("nx must bn integer")

if nx < 1:
    raise ValueError("nx must be a positive integer")

#innitialize weights with a randon nomarl distribution
self.W = np.random.rand(1, nx)

#initialize bias and activated output to 0
self.b = 0
self.A = 0