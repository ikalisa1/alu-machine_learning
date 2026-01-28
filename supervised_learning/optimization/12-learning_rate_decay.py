#!/usr/bin/env python3
"""Module for learning rate decay in TensorFlow."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Create a learning rate decay operation using inverse time decay.

    Args:
        alpha: The original learning rate
        decay_rate: The weight used to determine the rate at which alpha decays
        global_step: The global step variable
        decay_step: The number of passes of gradient descent that should occur
                   before alpha is decayed further

    Returns:
        The learning rate decay operation
    """
    decay_count = tf.floordiv(global_step, decay_step)
    alpha_decayed = alpha / (1.0 + decay_rate *
                             tf.cast(decay_count, tf.float32))
    return alpha_decayed
