import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class DataSource(ABC):
    """Abstract class to define the interface for the data source"""
    @abstractmethod
    def fetch_data(self):
        """Retrieve gross data from the data source"""
        pass

class ExcelDataSource(DataSource):
    """Class to fetch data from an Excel file"""
    def __init__(self, file_path:str="\\data\\data.xlsx", sheet_name:str="data", index_col:int=0):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.index_col = index_col

    def fetch_data(self):
        """Retrieve gross data from excel file"""
        data = pd.read_excel(self.file_path, sheet_name=self.sheet_name, index_col=self.index_col)
        # Convert excel date to datetime
        data.index = pd.to_datetime("1899-12-30") + pd.to_timedelta(data.index, unit="D")
        return data
    
    def fetch_data_core(self):
        """Retrieve gross data from excel file"""
        data = pd.read_excel(self.file_path, sheet_name=self.sheet_name, index_col=self.index_col)
        data.index =pd.to_datetime(data.index)
        return data


class CSVDataSource(DataSource):
    """Class to fetch data from a CSV file"""
    def __init__(self, file_path:str="//data//data.csv", index_col:int=0, date_column=None):
        self.file_path = file_path
        self.index_col = index_col
        self.date_column = date_column

    def fetch_data(self):
        """Retrieve gross data from csv file"""
        data = pd.read_csv(self.file_path, index_col=self.index_col)
        if self.date_column:
            data.index = pd.to_datetime(data.index)
        return data

class DataManager:
    """Class to manage, clean and preprocess data"""
    def __init__(self, data_source:DataSource, max_consecutive_nan:int=5):
        self.data_source = data_source
        self.max_consecutive_nan = max_consecutive_nan
        self.raw_data = None
        self.cleaned_data = None
        self.returns = None

    def load_data(self):
        """Load data from the data source"""
        self.raw_data = self.data_source.fetch_data()
        return self.raw_data

    def clean_data(self):
        """Clean the data by filling missing values"""
        if self.raw_data is None:
            self.load_data()

        df_filled = self.raw_data.copy()

        for col in self.raw_data.columns:
            series = self.raw_data[col]
            first_valid_index = series.first_valid_index()

            if first_valid_index is None:
                continue

            is_nan = series.isna()
            counter = 0

            for i in series.index:
                if i < first_valid_index:
                    continue

                if is_nan[i]:
                    counter += 1
                    if counter <= self.max_consecutive_nan:
                        i_idx = series.index.get_loc(i)
                        df_filled.iloc[i_idx, self.raw_data.columns.get_loc(col)] = df_filled.iloc[
                            i_idx - 1, self.raw_data.columns.get_loc(col)]

                else:
                    counter = 0

        self.cleaned_data = df_filled
        return self.cleaned_data

    def compute_returns(self):
        """Compute returns from the cleaned data"""
        if self.cleaned_data is None :
            self.clean_data()

        self.returns = self.cleaned_data.pct_change(fill_method=None)
        #self.returns = self.returns.iloc[1:, :]
        return self.returns

    def get_data(self):
        """Get all data prepared"""
        if self.returns is None:
            self.compute_returns()

        return {'raw_data' : self.raw_data,
                'cleaned_data' : self.cleaned_data,
                'returns' : self.returns
                }
    
    def align_dataframes(self, benchmark_df: pd.DataFrame, universe_df: pd.DataFrame):
        """
        Align two DataFrames based on common dates in their indexes.
        """
        common_dates = benchmark_df.index.intersection(universe_df.index)
        benchmark_aligned = benchmark_df.loc[common_dates]
        universe_aligned = universe_df.loc[common_dates]
        return benchmark_aligned, universe_aligned


class BenchmarkDataSource(DataSource):
    """Class to fetch benchmark data from a formatted Excel file."""
    def __init__(self, file_path: str, date_columns: list, value_columns: list):
        self.file_path = file_path
        self.date_columns = date_columns
        self.value_columns = value_columns

    def fetch_data(self):
        """Fetch and format benchmark data from the Excel file."""
  
        data = pd.read_excel(self.file_path, sheet_name=0, header=None)
        column_names = data.iloc[5].dropna().values # Retrieve the fund names (assumed to be on the 6th row, index 5) 
        data_values = data.iloc[6:].reset_index(drop=True) # Extract the data starting from the 7th row (index 6)
        formatted_data = pd.DataFrame()

        # Process each pair (date column, value column)
        for i, (date_col, val_col) in enumerate(zip(self.date_columns, self.value_columns)):
            temp_df = data_values[[date_col, val_col]].dropna()
            temp_df.columns = ["Date", column_names[i]]
            temp_df = temp_df.iloc[1:, :]
            temp_df["Date"] = pd.to_datetime(temp_df["Date"], errors='coerce')
            temp_df = temp_df.set_index("Date")

            # Merge the values into a single DataFrame
            if formatted_data.empty:
                formatted_data = temp_df
            else:
                formatted_data = formatted_data.join(temp_df, how="outer")
        
        # Sort by date and drop missing values
        formatted_data = formatted_data.sort_index().dropna()
        return formatted_data
    
