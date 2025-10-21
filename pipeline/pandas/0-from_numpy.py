#!/usr/bin/env python3
import pandas as pd


def from_numpy(array):
    """Label the columns of a DataFrame alphabetically.

        Args:
            array (pandas.DataFrame): The input DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with alphabetically labeled columns.
        """
    rows = array.shape[1]
    columns = [chr(i) for i in range(65, 65 + rows)]  # ascii value manipulation
    df = pd.DataFrame(array, columns=columns)
    return df