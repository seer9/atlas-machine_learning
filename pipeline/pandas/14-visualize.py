#!/usr/bin/env python3
"""visualize task"""
import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__('2-from_file').from_file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# preprocessing
df = df.drop(columns=['Weighted_Price'])
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# filter data from 2017 and beyond
df = df[df['Date'] >= '2017-01-01']
df = df.set_index('Date')

# fill missing values
df["Close"].fillna(method='ffill', inplace=True)
df["Open"].fillna(value=df['Close'].shift(1, fill_value=0), inplace=True)
df["High"].fillna(value=df['Close'].shift(1, fill_value=0), inplace=True)
df["Low"].fillna(value=df['Close'].shift(1, fill_value=0), inplace=True)
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)

# Resample data at daily intervals and aggregate
df = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
})

# Plot all data on one chart
plt.figure(figsize=(8, 6))

columns = ['High', 'Low', 'Open', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']
for column in columns:
    plt.plot(df.index, df[column], label=column)

plt.xlabel('Date')
plt.legend()
plt.show()
