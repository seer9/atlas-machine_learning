#!/usr/bin/env python3
"""
builds, trains, and saves a neural network classifier
"""
import tensorflow.compat.v1 as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    param X_train: numpy.ndarray of shape (m, 784)
    param Y_train: one-hot numpy.ndarray of shape (m, 10)
    param X_valid: numpy.ndarray of shape (m, 784)
    param Y_valid: numpy.ndarray of shape (m, 10)
    param layer_sizes: the number of nodes in each layer of the network
    param activations: the activation functions for each layer of the network
    param alpha: the learning rate
    param iterations: the number of iterations to train over
    param save_path: designates where to save the model
    return: the path where the model was saved
    """
    m, nx = X_train.shape
    _, classes = Y_train.shape
    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations + 1):
            loss_t, accuracy_t = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            loss_v, accuracy_v = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(loss_t))
                print("\tTraining Accuracy: {}".format(accuracy_t))
                print("\tValidation Cost: {}".format(loss_v))
                print("\tValidation Accuracy: {}".format(accuracy_v))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        return saver.save(sess, save_path)
