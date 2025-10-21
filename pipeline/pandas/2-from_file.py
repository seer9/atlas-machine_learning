#!/usr/bin/env python3
""" from_file task"""
import pandas as pd


def from_file(filename, delimiter):
    """Load a DataFrame from a file with a specified delimiter.

    Args:
        filename: The path to the file
        delimiter: The delimiter used in the file

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
