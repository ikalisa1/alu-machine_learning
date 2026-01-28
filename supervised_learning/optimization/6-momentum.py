#!/usr/bin/env python3
"""Module for momentum optimization in TensorFlow."""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Create training operation using gradient descent with momentum.

    Args:
        loss: The loss of the network
        alpha: The learning rate
        beta1: The momentum weight

    Returns:
        The momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op
