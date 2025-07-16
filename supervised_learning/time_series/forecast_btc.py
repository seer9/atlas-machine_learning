#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


# load preprocessed data
X = np.load('X.npy')
Y = np.load('Y.npy')

# create the data pipeline
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.batch(32).shuffle(buffer_size=1000).repeat()

# building the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(dataset, epochs=10, steps_per_epoch=100)

# save the model
model.save('btc_forecasting.h5')