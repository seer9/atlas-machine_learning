#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(
        file=('/supervised_learning/time_series/coinbase.csv'),
        window_size=24):
    
    ds = pd.read_csv(file)

    # print column
    print("Columns in dataset:", ds.columns)
    if 'Close' in ds.columns:
        ds = ds[['Close']]
    elif:
        ds = ds[['close']]
    else:
        raise ValueError("Close or close column not found in dataset")
    
    # drop NaN values
    ds = ds.dropna()

    # normalize the data
    scaler = MinMaxScaler()
    ds_scaled = scaler.fit_transform(ds)

    # create windows
    X, y = [], []
    for i in range(len(ds_scaled) - window_size):
        X.append(ds_scaled[i:i + window_size])
        y.append(ds_scaled[i + window_size])

    # save
    np.save('X.npy', np.array(X))
    np.save('y.npy', np.array(y))
    np.save('scaler.npy', scaler)

    if __name__ == "__main__":
        preprocess_data()
