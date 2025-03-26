#!/usr/bin/env python3
""" model building with transfer learning """
import tensorflow.keras as K


def preprocess_data(X, Y):
    """ preprocesses the data"""
    X_p = K.applications.undecided.preprocess_input(X) # tbt
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    preprocess_data()

# build model with transfer learning
# load the model
# evaluate the model
# save the model

