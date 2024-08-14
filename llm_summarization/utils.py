#utils.py


import pandas as pd
from llm_summarization.config import Config

def text_processor(input_df):
    '''
    Performs text processing and data cleaning on the input DataFrame

    Args:
        input_df (pd.DataFrame): A DataFrame including the abstract and title columns where each row corrosponds to a different article

    Returns:
        pd.DataFrame: A pre-processed DataFrame
    '''
    pre_processed_df = pd.concat([
        (input_df[Config.COL1].str.replace('\n', ' ')).to_frame(),
        (input_df[Config.COL2].str.replace('\n', ' ')).to_frame()
    ], axis=1)

    return pre_processed_df