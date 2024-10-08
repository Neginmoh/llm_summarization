#config.py


import os

class Config:
    VERBOSE = True  # if True additional messages will be printed

    MODEL_MAX_LENGTH = 512  # maximum number of tokens for the model input
    MAX_NEW_TOKENS = 512  # maximum number of new generated tokens
    TEMPERATURE = 0.001  # controls the randomness of the generated outputs
    TOP_K = 10  # k most probable tokens are selected at each step
    TASK = "text-generation"  # task performed by LLM model
    DEVICE_MAP = "auto"  # select appropriate available devices
    DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # pretrained LLM Model
    TRUNCATION = True  # if True truncate the input to a maximum length specified by the max_length argument or the model_max_length if no max_length is provided 
    RETURN_FULL_TEXT=False  # if False only the new output is returned
    DO_SAMPLE=True  # if True multinomial sampling is performed

    COL1, COL2, COL3 = 'title', 'abstract', 'summary'  # COL1 and COL2 are input dataset columns used for text summarization; COL3 is the new output column
    MAX_CHUNK_COUNT = 10  # number of chunks (batches) to read from dataset each with the size of DATA_CHUNKSIZE
    DATA_CHUNKSIZE = 32  # number of lines to read per chunk (batch) at a time when reading the JSON file

    PROMPT_TEMPLATE =  """
              Write a very short summary of the text delimited by triple backticks, with the title delimited by triple dashes.
              Make sure that the length of the generated summary is only one sentence and it includes the key points of the text and title.
              title:
              ---{title}---
              
              text:
              ```{text}```
              
              Summary:
           """  # prompt template including the instruction for the LLM model based on title and abstract of dataset

    BASE_DIR = os.getcwd()  # base directory
    DATA_DIR = os.path.join(BASE_DIR, 'data')  # data directory
    OUTPUT_DIR = os.path.join(DATA_DIR, 'output')  # output directory within data directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # create a output directory if does not exist within data directory
    CLEAN_DATASET_PATH = os.path.join(OUTPUT_DIR,"clean_dataset.csv")  # path to the cleaned and processed dataset within output directory
    OUTPUT_DATASET_PATH =  os.path.join(OUTPUT_DIR,'output_dataset.csv')  # path to the final output dataset containing the generated summaries
