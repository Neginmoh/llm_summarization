# ArXiv Abstract Summarization with Llama-2
## Project Description

This project involves performing a Natural Language Processing (NLP) summarization task on the abstract of scientific articles available at ArXiv by using a pre-trained Llama-2 LLM model. The project aims to automate the text summarization task by utilizing a HuggingFace pipeline, generating short and concise summaries including the key points of each article, and assisting researchers to identify the important aspects of each article quickly. 

## Download Input Data

This package utilizes ArXiv dataset of scholarly articles:
1. Download this dataset from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download).
2. Extract and save this dataset in Json format to a known directory.

## Setup

1. **Llama-2 License agreement:**

To gain access to Llama-2 model visit [Meta website](https://llama.meta.com/llama-downloads/), review the Meta license agreement, and follow the required steps.
Make sure to enter the same email address as the HuggingFace account.

2. **HuggingFace Token:**

To use HuggingFace models, create an account, and obtain a HuggingFace token by following the steps on [HuggingFace website](https://huggingface.co/docs/hub/en/security-tokens).

After being granted access from Meta, request access to Llama-2 model on HuggingFace from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

3. **Requirements**

Install the required packages:

```python
pip install -r requirements.txt
```

## Usage

1. **HuggingFace login:**

Run the code below from the command line after replacing HF_TOKEN with the obtained token:

```python
huggingface-cli login --add-to-git-credential --token HF_TOKEN
```

2. **Run Inference:**

Run the code below from the command line after replacing INPUT_DATA_PATH with the path to the downloaded input dataset:

```python
python main.py INPUT_DATA_PATH
```

3. **Setting Variables:**

Most variables could be configured in the Config.py file.
Depending on the system you use and the available memory, you could change the MAX_CHUNK_COUNT corresponding to the number of batches processed from the original JSON dataset. Also, you might need to change the DATA_CHUNKSIZE corresponding to batch size, i.e., the number of data per batch.
Additionally, you can change MODEL_MAX_LENGTH and MAX_NEW_TOKENS to set the number of input and output tokens processed by the LLM model.

## Workflow

- Loads data to memory batch by batch from a JSON file.
- Performs data processing and data cleaning batch by batch and saves the cleaned data to a CSV file.
- Utilizes and configures a HuggingFace pipeline to streamline the NLP task workflow.
- Performs a text summarization batch by batch using a Llama-2 LLM model
- Saves the new DataFrame containing the cleaned input data and the generated summaries to a CSV file.

## Input and Output Format

### Input format

For an example of a single input see the [dataset page](https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download).
Note that, only the **title** and **abstract** will be used, for the text summarization task.

### Output format

Here is an example of a single cleaned output dataset (without summaries):
|   | **title**                | **abstract**                        |
|---|--------------------------|-------------------------------------|
| 0 | Calculation of prompt... | A fully differential calculation... |
| 1 | Sparsity-certifying Gra... | We describe a new algorithm...|

<br>

Here is an example of a single output dataset containing the summaries:
|   | **title**                | **abstract**                        | **summary** 	|
|---|--------------------------|-------------------------------------|-------------	|
| 0 | Calculation of prompt... | A fully differential calculation... | In this study, the authors... |
| 1 | Sparsity-certifying Gra... | We describe a new algorithm... | In this paper, the authors... |

## License

MIT license. See the LICENSE file.