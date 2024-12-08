"""Implementation for data loading."""
import pandas as pd
import os

from dataset.data_processor import DataProcessor

class DataLoader:
    """Class to load dataset from csv file."""
    def __init__(self, fpath):
        """Initializer"""
        assert fpath.endswith(".csv"), "Only CSV file loading supported."

        assert os.path.exists(fpath), f"File not found: {fpath}"

        self.fpath = fpath
        self.data_frame = None

    def load_csv(self):
        """function to return loaded dataset from csv file"""
        self.data_frame = pd.read_csv(self.fpath)
        print ("[I] data loaded:", self.data_frame.shape)

    def clean(self, keep_columns=None, set_columns=None, plot_hist=None):
        proc = DataProcessor(self.data_frame)
        if keep_columns:
            proc.keep_columns(keep_columns)
        if set_columns:
            proc.set_columns(set_columns)
        proc.remove_nans()
        self.data_frame = proc.get_data()
        print ("[I] after cleaning data:", self.data_frame.shape)

    def plot_data(self, column):
        proc = DataProcessor(self.data_frame)
        proc.plot_hist(column)
