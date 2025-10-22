#!/usr/bin/env python3
"""hierarchy task"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Creates a hierarchical dataframe from two dataframes.
    Args:
        df1: first dataframe
        df2: second dataframe
    Return:
        the hierarchical dataframe
    """
    df1 = df1.set_index('Timestamp')
    df2 = df2.set_index('Timestamp')

    # set the range of target data
    df2 = df2[(df2.index.get_level_values('Timestamp') >= 1417411980) &
              (df2.index.get_level_values('Timestamp') <= 1417417980)]
    df1 = df1[(df1.index.get_level_values('Timestamp') >= 1417411980) &
              (df1.index.get_level_values('Timestamp') <= 1417417980)]

    # concat the dataframes with keys
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    # timestamp as the first level
    df = df.swaplevel(0, 1)
    # sort the dataframe by index
    df = df.sort_index()

    return df
