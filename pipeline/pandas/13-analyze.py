#!/usr/bin/env python3
"""analyze task"""


def analyze(df):
    """
    analyze the dataframe.
    Args:
        df: input dataframe
    Returns:
        sum: statistics of the new dataframe.
    """
    drop = df.drop(columns=["Timestamp"], errors="ignore")
    sum = drop.describe()
    return sum
