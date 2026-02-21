#!/usr/bin/env python3
"""Module for creating forward propagation graph"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network

    Args:
        x: the placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer

    Returns:
        the prediction of the network in tensor form
    """
    layer = x
    for i in range(len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
