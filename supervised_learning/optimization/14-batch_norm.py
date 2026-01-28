#!/usr/bin/env python3
"""Module for batch normalization layer in TensorFlow."""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Create a batch normalization layer for a neural network in tensorflow.

    Args:
        prev: The activated output of the previous layer
        n: The number of nodes in the layer to be created
        activation: The activation function to use on the output

    Returns:
        A tensor of the activated output for the layer
    """
    # Create the dense layer
    dense = tf.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"
        )
    )
    Z = dense(prev)

    # Create gamma and beta as trainable variables
    gamma = tf.Variable(
        tf.ones([n]),
        trainable=True
    )
    beta = tf.Variable(
        tf.zeros([n]),
        trainable=True
    )

    # Calculate batch mean and variance
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Apply batch normalization
    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        beta,
        gamma,
        1e-8
    )

    # Apply activation function
    return activation(Z_norm)
