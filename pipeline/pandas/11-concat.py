#!/usr/bin/env python3
"""concat task"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two dataframes and sorts them by index.
    Args:
        df1: first dataframe.
        df2: second dataframe.
    Return:
        the concatenated dataframe sorted by index.
    """
    df1 = index(df1)
    df2 = index(df2)
    df = pd.concat([df1, df2])
    df = df.sort_index()
    return df
