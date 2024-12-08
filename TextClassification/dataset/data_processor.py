"""Implementation to process data-frame"""
import pandas as pd
import os

class DataProcessor:
    """Class to process data-frame"""

    def __init__(self, data_frame):
        """Initializer"""
        self.data_frame = data_frame

    def keep_columns(self, columns):
        """function to keep only specific columns in data-frame"""
        self.data_frame = self.data_frame[columns]

    def remove_nans(self):
        """function to remove NaN from data-frame"""
        self.data_frame = self.data_frame.dropna()

    def plot_hist(self, column, path="outputs/plot"):
        """functiont to plot data histogram."""
        plot = self.data_frame[column].value_counts(normalize = True).plot.bar()
        os.makedirs(path, exist_ok=True)
        fig = plot.get_figure()
        fig.savefig(f"{path}/output.png")

    def set_columns(self, columns):
        """function to set data-frame columns"""
        self.data_frame.columns = columns

    def get_data(self):
        """function to return data-frame"""
        return self.data_frame
