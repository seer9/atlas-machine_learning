#!/usr/bin/env python3
"""
removes entries where Close has NaN values.
Arg:
    df: The input DataFrame.
Returns: The modified DataFrame.
"""
def prune(df):
    """Removes entries where Close has NaN values."""
    df = df.dropna(subset=['Close'])
    return df
