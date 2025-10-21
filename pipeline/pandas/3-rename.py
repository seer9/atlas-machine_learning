#!/usr/bin/env python3
"""rename task"""
import pandas as pd


def rename(df):
    """takes a pd.DataFrame as input.
    Arg:
        df: The input DataFrame.
    Returns: the modified DataFrame with renamed columns.
    """
    # Rename timestamp
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

    # rename Datetime to datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    print(df[['Datetime', 'Close']].tail())
