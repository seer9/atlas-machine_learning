#!/usr/bin/env python3
"""builds a modified version of the LeNet-5 architecture using tensorflow"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    function that builds a version of the LeNet-5 architecture
    args:
        x: the input images for the network
        y: one-hot encoded labels of x
    returns:
        fc3: the networks prediction
        train_op: networks training operation
        loss: the loss
        accuracy: networks accuracy
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    act_fuct = tf.nn.relu

    conv1 = tf.layers.Conv2D(filters=6, kernel_size=5,
                             padding='same', activation=act_fuct,
                             kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(filters=16, kernel_size=5,
                             padding='valid', activation=act_fuct,
                             kernel_initializer=init)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    flatten = tf.layers.Flatten()(pool2)

    fc1 = tf.layers.Dense(units=120, activation=act_fuct,
                          kernel_initializer=init)(flatten)
    fc2 = tf.layers.Dense(units=84, activation=act_fuct,
                          kernel_initializer=init)(fc1)
    fc3 = tf.layers.Dense(units=10,
                          kernel_initializer=init)(fc2)

    loss = tf.losses.softmax_cross_entropy(y, fc3)
    adam_op = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(fc3, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    softmax = tf.nn.softmax(fc3)

    return softmax, adam_op, loss, acc
