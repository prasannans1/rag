#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
#

import os
from typing import Generator, List, Tuple
from datasets import Dataset
import json
import random
from ruamel.yaml import CommentedMap
import glob

random.seed(42)
# Define a function to return a generator of label and text pairs from the JSON lines
def yield_data(filename: str) -> Generator[Tuple[int, str], None, None]:
    """Generator of tuples of label, text

    Args:
        filename (str): input file

    Yields:
        Generator[Tuple[int, str], None, None]: Generator of tuples of label and text
    """
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            # if label in json is "Biology", consider label as 0
            label = 0
            if "Physics" == data["label"]:
                label = 1
            elif "History" == data["label"]:
                label = 2
                
            yield label, data["text"]

# Get the train and test lists from the input json file
def get_train_test_lists(cfg: CommentedMap) -> Tuple[List[Tuple[int, str]]]:
    """Get the train and test lists

    Args:
        cfg (CommentedMap): configuration read from config file. 

    Returns:
        Tuple[List[Tuple[int, str]]]: The train and test lists of tuples containing label and text
    """
    books_dir = cfg["paths"]["books_dir"]

    # Use glob to find all JSONL files in the chunks directory
    jsonl_files = glob.glob(os.path.join(f"{books_dir}/chunks", "*.json"))

    # List comprehension to collect examples from all JSONL files
    examples = [(label, text) for jsonl_file in jsonl_files for label, text in yield_data(jsonl_file)]
    examples = [example for example in examples if len(example[1]) > 100]
    random.shuffle(examples)
    train_size = int(len(examples) * 0.8)
    train_list = examples[:train_size]
    test_list = examples[train_size:]
    return train_list, test_list

def tuples_list_to_dataset(tuples_list: List[Tuple[int, str]]) -> Dataset:
    """Converts list of tuples of label, text to Hugging face dataset

    Args:
        tuples_list (List[Tuple[int, str]]): A list of tuples containing subject label along with the text

    Returns:
        Dataset: The hugging face dataset with columns "label" and "text" with these two values
    """
    # Separate the labels and texts
    labels, texts = zip(*tuples_list)
    # Create a dictionary
    data_dict = {"label": labels, "text": texts}
    # Convert to Dataset
    return Dataset.from_dict(data_dict)

# Define the tokenizer function
def tokenize_function(examples, tokenizer):
    """Tokenize using the tokenizer defined.

    Args:
        examples (_type_): dictionary containing text key
        tokenizer (_type_): The tokenizer

    Returns:
        _type_: The tokenized list returned
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# The prediction using the trained model on a piece of text - it outputs the corresponding subject   
def predict(model, tokenizer, text, device) :
    """Model's predictions on an input text

    Args:
        model (_type_): The trained model
        tokenizer (_type_): The tokenizer
        text (_type_): Text that needs to be classified
        device (str): cpu or cuda

    Returns:
        _type_: The predicted subject
    """
    tokens = tokenizer([text], padding="max_length", truncation=True, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    model = model.to(device)
    output = model(**tokens)
    label = output.logits.argmax(1).item()
    
    subject = 'Biology' # if label is 0, subject is 'Biology'
    if label == 1:
        subject = 'Physics'
    elif label == 2:
        subject = 'History'
        
    return subject

def get_latest_directory(path: str) -> str|None:
    """Get the latest checkpoint folder from the path

    Args:
        path (str): parent path

    Returns:
        str|None: path of latest updated directory
    """
    # Get a list of all first-level directories in the specified path
    directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    # If no directories found, return None
    if not directories:
        return None
    
    # Find the directory with the latest modification time
    latest_dir = max(directories, key=os.path.getmtime)
    
    return latest_dir

# Some sample predictions to validate model output
def sample_predictions(model, tokenizer, device):
    """Evaluate sample predictions

    Args:
        model (_type_): Trained model
        tokenizer (_type_): Tokenizer
        device (str): cpu or cuda
    """
    text1 = "The rate of change of displacement is velocity"
    print(f"text: {text1} subject: {predict(model, tokenizer, text1, device)}")
    text2 = "Kidney plays an important role in purifying blood"
    print(f"text: {text2} subject: {predict(model, tokenizer, text2, device)}")
    text3 = "Many countries obtained their freedom by 1950"
    print(f"text: {text3} subject: {predict(model, tokenizer, text3, device)}")