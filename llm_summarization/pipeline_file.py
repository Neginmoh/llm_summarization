# llm_summarization/pipeline_file.py


import torch
from transformers import pipeline
from llm_summarization.model import ModelManage
from llm_summarization.config import Config
from llm_summarization.prompt import prompt_processor


class Pipeliner:
    '''
    Initialize and configures a pipeline to streamline the workflow and then uses the pipeline to the specified task (summarization) on the provided document.
    '''
    def __init__(self, model_name):
        '''
        Initialize the HuggingFace pipeline

        The self.task performed by pipeline is set according to Config.TASK, and self.model_name is obtained from the input provided to Pipeliner instance.
        An instance of ModelManege is created using self.model_name, and the pipeline tokenizer is assigned from ModelManage instance.
   
        Args:
            model_name (str) : The name of the LLM model to be used
        '''
        self.task = Config.TASK
        self.model_name = model_name
        self.model_manage = ModelManage(self.model_name)
        self.tokenizer = self.model_manage.tokenizer    
        self.pipe = pipeline(self.task,
                        model=self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=Config.DEVICE_MAP)
    
    def get_summary(self, docu_info):
        '''
        Generates a summary based on provided document information

        Args:
            docu_info (list): a list containing:
                1. text (str): The abstract of the article to be summarized
                2. title (str): The title of the article

        Returns:
            str: The generated summary
        '''
        prompt = prompt_processor(docu_info)
        sequences = self.pipe(prompt,
                        do_sample=Config.DO_SAMPLE,
                        top_k=Config.TOP_K,
                        truncation=Config.TRUNCATION,
                        # num_return_sequences=1,
                        max_new_tokens=Config.MAX_NEW_TOKENS, 
                        # max_length = 512,
                        temperature=Config.TEMPERATURE,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_full_text=Config.RETURN_FULL_TEXT)
        return sequences[0]['generated_text']  