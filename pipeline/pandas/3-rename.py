#!/usr/bin/env python3
"""rename task"""
import pandas as pd


def rename(df):
    """takes a pd.DataFrame as input.
    Arg:
        df: The input DataFrame.
    Returns: the modified DataFrame with renamed columns.
    """
    from_file = __import__('2-from_file').from_file

    # dataset
    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

    # Rename timestamp
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

    # rename Datetime to datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    print(df[['Datetime', 'Close']].tail())
    