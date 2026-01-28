#!/usr/bin/env python3
"""Mini-batch training function."""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Train a loaded neural network model using mini-batch gradient descent.

    Args:
        X_train (np.ndarray): Training data of shape (m, 784).
        Y_train (np.ndarray): Training labels (one-hot) of shape (m, 10).
        X_valid (np.ndarray): Validation data of shape (m, 784).
        Y_valid (np.ndarray): Validation labels (one-hot) of shape (m, 10).
        batch_size (int): Number of data points in a batch.
        epochs (int): Number of times to pass through the dataset.
        load_path (str): Path from which to load the model.
        save_path (str): Path to save the model after training.

    Returns:
        str: The path where the model was saved.
    """
    with tf.Session() as sess:
        # Import meta graph and restore session
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        # Get tensors and ops from collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # Calculate and print metrics before training
        train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sess.run(accuracy,
                                  feed_dict={x: X_train, y: Y_train})
        valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuracy = sess.run(accuracy,
                                  feed_dict={x: X_valid, y: Y_valid})

        print("After 0 epochs:")
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

            # Mini-batch loop
            m = X_train.shape[0]
            num_batches = (m + batch_size - 1) // batch_size

            for step in range(num_batches):
                start = step * batch_size
                end = min(start + batch_size, m)

                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                # Train on mini-batch
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                # Print step metrics every 100 steps (after step, not before)
                if (step + 1) % 100 == 0:
                    step_cost = sess.run(loss,
                                        feed_dict={x: X_batch, y: Y_batch})
                    step_accuracy = sess.run(accuracy,
                                            feed_dict={x: X_batch,
                                                      y: Y_batch})
                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

            # Calculate and print metrics after each epoch
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy,
                                     feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy,
                                     feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the session
        save_path = saver.save(sess, save_path)

    return save_path
