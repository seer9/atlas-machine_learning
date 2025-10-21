#!/usr/bin/env python3
"""array task"""


def array(df):
    """takes a pd.DataFrame as input. select last 10 rows of the
    high and close columns. Convert the resulting DataFrame to a np.array.
    Arg:
        df: The input DataFrame.
    Returns: np.array.
    """
    # Select last 10 rows of 'High' and 'Close' columns
    selected_data = df[['High', 'Close']].tail(10)

    # Convert the resulting DataFrame to a np.array
    result_array = selected_data.to_numpy()

    return result_array
