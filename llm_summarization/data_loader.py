#data_loader.py


import pandas as pd
import os
from llm_summarization.config import Config

class DataLoader:
    '''
    This class loads large datasets in CSV and JSON files in chunks
    '''
    def __init__(self, input_path):
        '''
        Initialize the DataLoader instance with:
            - self.main_dataset_path: path to input dataset given by input_path
            - self.clean_dataset_path: path to pre-processed dataset
            - self.data_chunksize: chunksize given by Config.DATA_CHUNKSIZE

        Args:
            input_path (str): path to input dataset 
        '''
        self.clean_dataset_path = Config.CLEAN_DATASET_PATH
        self.main_dataset_path = input_path
        self.data_chunksize = Config.DATA_CHUNKSIZE

    def csv_loader(self):
        '''
        Reads pre-processed CSV dataset chunk by chunk from self.clean_dataset_path

        Yields:
            pd.DataFrame: A chunk of the CSV data in DataFrame format
        '''
        for chunk in pd.read_csv(self.clean_dataset_path, lines=True, chunksize=self.data_chunksize):
            yield chunk
    
    def json_loader(self):
        '''
        Reads the input JSON dataset chunk by chunk from self.main_dataset_path

        Raises:
            FileNotFoundError: If the input data is not found at the given self.main_dataset_path 
        Yields:
            pd.DataFrame: A chunk of the JSON data in DataFrame format
        '''
        if not os.path.isfile(self.main_dataset_path):
            raise FileNotFoundError(f"The input data is not found at {self.main_dataset_path}")
        for chunk in pd.read_json(self.main_dataset_path, lines=True, chunksize=self.data_chunksize):
            yield chunk