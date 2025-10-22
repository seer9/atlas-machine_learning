#!/usr/bin/env python3
"""indexing task"""


def index(df):
    """
    Sets the Timestamp column as the index pf the datafrmae.
    Args:
        df: dataframe to be modified.
    Return:
        the modified dataframe.
    """
    df = df.set_index('Timestamp')
    return df
