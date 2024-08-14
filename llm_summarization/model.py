# llm_summarization/model.py


from transformers import AutoTokenizer
from llm_summarization.config import Config


class ModelManage:
    '''
    This class handles the model and tokenizer configuration
    '''
    def __init__(self, model_name):
        '''
        Initialize the model, model configuration, and tokenizer

        Args:
            model_name (str): The name of the LLM model to be used
        '''
        self.model_max_length = Config.MODEL_MAX_LENGTH
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=self.model_max_length)