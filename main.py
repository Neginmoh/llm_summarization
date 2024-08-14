# main.py


import sys
import os
import argparse

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_summarization.summarizer import SummarizerLLM

def main():
    '''
    The main function which parses the input dataset path from command line and runs the inference to generate summaries.

    Steps:
    1. Parses the command line argument for the input dataset path.
    2. Creates an instance of SummarizerLLM instance with the input dataset path.
    3. Executes the inference process by calling run_inference() method on the instance.

    Args:
        None
    Returns:
        None
    Raises:
        FileNotFoundError: If the input dataset file is not found at the provided path.
        KeyboardInterrupt: If the program interrupted by the user.
    '''
    try:
        parser = argparse.ArgumentParser(description='Enter the input dataset path')
        parser.add_argument('path')
        args = parser.parse_args()
        input_path = args.path

        sum_inst = SummarizerLLM(input_path)
        sum_inst.run_inference()

    except FileNotFoundError as e:
        print(f'Error : {e}')
        sys.exit(0)

    except KeyboardInterrupt:
        print(f'Error : KeyboardInterrupt')
        sys.exit(0)

    finally:
        print('Program execution has ended')

if __name__ == "__main__":
    main()