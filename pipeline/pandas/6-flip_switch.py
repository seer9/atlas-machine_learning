#!/usr/bin/env python3
"""flip and switch it"""


def flip_switch(df):
    """flips the dataframe upside down and switches the first column with the last one"""
    df = df.sort_index(ascending=False)
    df = df.transpose()
    return df
