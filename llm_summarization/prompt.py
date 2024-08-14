# prompt.py


from llm_summarization.config import Config

def prompt_processor(docu_info):
    '''
    This function creates a prompt based on the document's contents and the prompt template provided in Config.PROMPT_TEMPLATE.

    Args:
        docu_info (list): a list containing:
            1. text (str): The abstract of the article to be summarized
            2. title (str): The title of the article
    Returns:
        str: The formatted prompt to instruct the LLM model
    '''
    prompt_template = Config.PROMPT_TEMPLATE
    prompt = prompt_template.format(title=docu_info[0], text=docu_info[1])
    
    return prompt
  