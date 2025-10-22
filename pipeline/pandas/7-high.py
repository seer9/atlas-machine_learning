#!/usr/bin/env python3
"""sorting task"""


def high(df):
    """Sorts by high price in descending order.
    Arg:
        df: The input DataFrame.
    Returns: The sorted DataFrame.
    """
    df = df.sort_values(by='High', ascending=False)
    return df
