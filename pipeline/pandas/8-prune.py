#!/usr/bin/env python3
"""prune task"""


def prune(df):
    """
    removes entries where Close has NaN values.
    Arg:
        df: The input DataFrame.
    Returns: The modified DataFrame.
    """
    df = df.dropna(subset=['Close'])
    return df
