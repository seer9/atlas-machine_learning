#!/usr/bin/env python3
""" from_numpy task"""
import pandas as pd


def from_numpy(array):
    """Label the columns of a DataFrame alphabetically.

        Args:
            array: The input DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with proper labels.
        """
    rows = array.shape[1]
    # ascii value manipulation
    columns = [chr(i) for i in range(65, 65 + rows)]
    df = pd.DataFrame(array, columns=columns)
    return df
