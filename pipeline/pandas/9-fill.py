#!/usr/bin/env python3
"""fill task"""


def fill(df):
    """
    fills NaN values in the 'Close' column using forward fill method.
    fillls vlaues in for high, low and open as well.
    sets missing values in Volume_BTC and Volume_USD to 0.
    Arg:
        df: The input DataFrame.
    Returns: The modified DataFrame.
    """
    df = df.drop(columns=['Weighted_Price'])
    df['Close'].fillna(method='ffill', inplace=True)
    df['High'].fillna(value=df['Close'].shift(1, fill_value=0), inplace=True)
    df['Low'].fillna(value=df['Close'].shift(1, fill_value=0), inplace=True)
    df['Open'].fillna(value=df['Close'].shift(1, fill_value=0), inplace=True)
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
