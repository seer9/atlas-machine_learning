#!/usr/bin/env python3
"""slice task"""


def slice(df):
    """takes a pd.DataFrame as input.
    Arg:
        df: The input DataFrame.
    Returns: columns 'High', 'Low', 'Close', and 'Volume_BTC',
             selecting every 60th row.
    """

    # Select every 60th row and desired columns
    df = df.iloc[::60, 2:6]

    return df
