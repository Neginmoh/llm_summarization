# llm_summarization/summarizer.py


from llm_summarization.config import Config
from llm_summarization.model import ModelManage
from llm_summarization.pipeline_file import Pipeliner
from llm_summarization.data_loader import DataLoader
from llm_summarization.prompt import prompt_processor
from llm_summarization.utils import text_processor
import pandas as pd

class SummarizerLLM:
    '''
    This class performs an end-to-end summarization task using a LLM model
    '''

    def __init__(self, input_path):
        '''
        Initializes the instance variables.

        self.model_name is set to Config.DEFAULT_MODEL which provides the LLM model to be used for NLP task.
        self.pipe_line is an instance of the Pipeliner class to streamline the summarization process, using the LLM model provided in self.model_name.
        self.data_loading is an instance of the DataLoader class for loading a file from input_path.
        self.data_loading.json_loader() is a generator object that yields data batches from a JSON file located at input_path.
        self.max_chunk_count set to Config.MAX_CHUNK_COUNT, specifies how many batches we want to process and obtain their summaries
        self.data_chunksize is set to Config.DATA_CHUNKSIZE specifies the size of each batch of data.
        self.clean_dataset_path is set to Config.CLEAN_DATASET_PATH, specifying the path to CSV file containing the cleaned DataFrame
        self.output_dataset_path = Config.OUTPUT_DATASET_PATH, specifying the path to CSV file containing the DataFrame of cleaned documents and their generated summaries.
        self.col1 is set to Config.COL1, which is the title column which will be used to summarize abstracts.
        self.col2 is set to Config.COL2, which is the abstract column that will be summarized.
        self.col3 is set to Config.COL3, which is the summary column containing the generated summaries.
        '''
        self.model_name = Config.DEFAULT_MODEL
        self.pipe_line = Pipeliner(self.model_name)
        self.data_loading = DataLoader(input_path)
        self.data_generator = self.data_loading.json_loader()
        self.max_chunk_count = Config.MAX_CHUNK_COUNT
        self.data_chunksize = Config.DATA_CHUNKSIZE
        self.clean_dataset_path = Config.CLEAN_DATASET_PATH
        self.output_dataset_path = Config.OUTPUT_DATASET_PATH
        self.col1 = Config.COL1
        self.col2 = Config.COL2
        self.col3 = Config.COL3
        
    def get_summary_list(self, pre_processed_batch, verbose=False):
        '''
        Generates summaries of the pre-processed batch of articles containing abstract and title for each row of the given DateFrame.
        The method loops through every document in the pre-processed batch, obtains its summary and handles Exception errors.

        Args:
            pre_processed_batch (pd.DataFrame): A DateFrame of pre-processed batch that includes the abstract and title columns where each row corresponds to a different article
            verbose (bool, optional): Disables/enables printing additional information, default is set to False (disable)
        Returns:
            list: A list of generated summaries corresponding to articles of pre_processed_batch
        Raises:
            Exception: Logs any other exceptions that occur during process, and 
                assigns 'No summary' as the summary for that document.
        '''
        if Config.VERBOSE == True:
            verbose = True

        summary_list=[]
        for i in range(len(pre_processed_batch)): 
            title = pre_processed_batch.iloc[i][self.col1]
            text = pre_processed_batch.iloc[i][self.col2]
            docu_info = [title, text]
             
            try:
                summary = self.pipe_line.get_summary(docu_info)
                if verbose:
                    print(f"Article #{i}; summary is generated")

            except Exception as e:
                if verbose:
                    print(f"Article #{i}; Error: {e}")
                summary = 'No summary'

            summary = summary.replace('\n',' ').strip()
            summary_list.append(summary)

        return summary_list

    def pre_process(self, batch, count):
        '''
        This method pre-processes the batch of data and saves the cleaned data to a specified file.
        Only self.col1 and self.col2 columns corresponding to the 'title' and 'abstract' will be kept.
        If this is the first batch, the file will be opened in write mode and the column names will be written to the CSV file located at self.clean_dataset_path
        For subsequent batches, the cleaned data is appended to the existing file.

        Args:
            batch (pd.DataFrame): A DataFrame containing the batch of data to be pre-processed
            count (int): the iteration count indicating which batch is being processed
        Returns:
            Tuple: a tuple containing:
                pd.DataFrame: The pre-processed batch of data
                int: the iteration count indicating which batch is pre-processed
        '''
        batch = batch[[self.col1, self.col2]]
        pre_processed_batch = text_processor(batch)
        if count == 0:
            mode = 'w' 
            header = True
        else:
            mode = 'a'
            header = False

        pre_processed_batch.to_csv(self.clean_dataset_path, mode=mode, header=header, index=False)
        
        return pre_processed_batch, count
    
    def post_process(self, summary_list, pre_processed_batch, count):
        '''
        This method adds a new column containing a list of summaries to the existing DataFrame and saves the DataFrame to a CSV file at specified path.
        The new column is self.col3 which corresponds to 'summary' and will be added to DataFrame with existing 'title' and 'abstract' columns.
        If this is the first batch, the file will be opened in write mode and the column names will be written to the CSV file at self.output_dataset_path
        For subsequent batches, the cleaned data is appended to the existing file.

        Args:
            summary_list (list): A list of generated summaries corresponding to articles of pre_processed_batch
            pre_processed_batch (pd.DataFrame): A DataFrame containing the batch of data that has been pre-processed 
            count (int): the iteration count indicating which batch is being post-processed
        Returns:
            int: the iteration count indicating which batch is post-processed
        '''
        if count == 0:
            mode = 'w' 
            header = True
        else:
            mode = 'a'
            header = False
        index_df = pre_processed_batch.index
        pre_processed_batch[self.col3] = pd.Series(summary_list, index=index_df)
        pre_processed_batch.to_csv(self.output_dataset_path, mode=mode, header=header, index=False)
        
        return count
        
    def run_inference(self, verbose=False):
        """
        This method runs the inference on data batches.
        It iterates over batches of data generated by self.data_generator.
        Performs pre-processing by self.pre_process method on batches, which also saves cleaned DataFrame to a file.
        Next, it generates their summaries by self.get_summary_list method
        Then performs post-processing by self.post_process method which adds the new column containing their summaries to the DataFrame and saves the DataFrame to another file.
        The process stops when the number of processed batches reaches the `self.max_chunk_count value.

        Args:
            verbose (bool, optional): If True prints extra messages. Default is False.
        Returns:
            None
        """
        if Config.VERBOSE == True:
            verbose = True

        count = 0
        for batch in self.data_generator:
            
            pre_processed_batch, count = self.pre_process(batch, count)
            summary_list = self.get_summary_list(pre_processed_batch, verbose)
            count = self.post_process(summary_list, pre_processed_batch, count)
            count += 1

            if verbose:
                print(f"batch num #{count-1}; summaries are saved \n")

            if count >= self.max_chunk_count:
                break

        
