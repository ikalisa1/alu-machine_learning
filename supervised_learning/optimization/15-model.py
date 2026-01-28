#!/usr/bin/env python3
"""Module for complete neural network model with batch normalization."""
import numpy as np
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Build, train, and save a neural network model with batch normalization.

    Args:
        Data_train: Tuple of (training inputs, training labels)
        Data_valid: Tuple of (validation inputs, validation labels)
        layers: List containing number of nodes in each layer
        activations: List of activation functions for each layer
        alpha: Learning rate
        beta1: Weight for first moment of Adam
        beta2: Weight for second moment of Adam
        epsilon: Small number to avoid division by zero
        decay_rate: Decay rate for inverse time decay
        batch_size: Mini-batch size
        epochs: Number of training epochs
        save_path: Path to save the model

    Returns:
        The path where the model was saved
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # Create placeholders
    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')
    is_training = tf.placeholder(tf.bool, name='is_training')

    # Build the network with batch normalization
    a = x
    for i, (layer_size, activation) in enumerate(zip(layers, activations)):
        # Dense layer
        z = tf.layers.dense(
            a,
            layer_size,
            activation=None,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"
            )
        )

        # Batch normalization (only if not the output layer)
        if i < len(layers) - 1:
            z = tf.layers.batch_normalization(
                z,
                training=is_training,
                epsilon=epsilon
            )

        # Activation function
        if activation is not None:
            a = activation(z)
        else:
            a = z

    y_pred = a

    # Loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y,
            logits=y_pred
        )
    )

    # Accuracy
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Optimizer with learning rate placeholder
    learning_rate = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )

    # Handle batch normalization updates
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs + 1):
            # Calculate learning rate with decay
            alpha_t = alpha / (1 + decay_rate * epoch)

            # Evaluate on full training and validation sets
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train,
                                                   is_training: False})
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train,
                                                      is_training: False})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid,
                                                   is_training: False})
            valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid,
                                                      is_training: False})

            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_acc))
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_acc))

            # Train on mini-batches if not at end
            if epoch < epochs:
                # Shuffle training data before each epoch
                perm = np.random.permutation(X_train.shape[0])
                X_train_shuffled = X_train[perm]
                Y_train_shuffled = Y_train[perm]

                step = 0
                for i in range(0, X_train.shape[0], batch_size):
                    X_batch = X_train_shuffled[i:i+batch_size]
                    Y_batch = Y_train_shuffled[i:i+batch_size]

                    _, batch_cost, batch_acc = sess.run(
                        [train_op, loss, accuracy],
                        feed_dict={
                            x: X_batch,
                            y: Y_batch,
                            learning_rate: alpha_t,
                            is_training: True
                        }
                    )

                    step += 1

                    if step % 100 == 0:
                        print('\tStep {}:'.format(step))
                        print('\t\tCost: {}'.format(batch_cost))
                        print('\t\tAccuracy {}'.format(batch_acc))

        save_path = saver.save(sess, save_path)

    return save_path
