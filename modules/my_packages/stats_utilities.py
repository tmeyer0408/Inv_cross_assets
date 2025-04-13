import pandas as pd
import numpy as np

class Statistics:

    @staticmethod
    def rolling_correlation(df: pd.DataFrame, col1: str, col2: str, window: int) -> pd.Series:
        """
        Computes the rolling correlation between two columns in a DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the time series data.
        - col1 (str): Name of the first column (e.g., strategy returns).
        - col2 (str): Name of the second column (e.g., benchmark returns).
        - window (int): Rolling window size.

        Returns:
        - pd.Series: Rolling correlation series.
        """
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"Columns '{col1}' and '{col2}' must be in the DataFrame.")

        return df[col1].rolling(window, min_periods=1).corr(df[col2])