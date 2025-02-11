#!/usr/bin/env python3
"""
evaluation of the output of a neural network
"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    param X: numpy.ndarray containing the input data to evaluate
    param Y: numpy.ndarray containing the one-hot labels for X
    param save_path: location to load the model from
    return: the network's prediction, accuracy, and loss
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        
        feed_dict = {x: X, y: Y}
        prediction = sess.run(y_pred, feed_dict)
        accuracy = sess.run(accuracy, feed_dict)
        loss = sess.run(loss, feed_dict)
        
        return prediction, accuracy, loss
