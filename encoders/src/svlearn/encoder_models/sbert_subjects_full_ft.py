#  ------------------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------------------

import itertools
import random
import time
import torch
from datasets import Dataset, concatenate_datasets
from sentence_transformers import (SentenceTransformer, 
                                   SentenceTransformerTrainer, 
                                   SentenceTransformerTrainingArguments, 
                                   losses, 
                                   evaluation)

from svlearn.config.configuration import ConfigurationMixin
from svlearn.util.hf_text_util import get_train_test_lists, tuples_list_to_dataset

def get_evaluator(test_dataset: Dataset, truncate_dim: int = None) -> evaluation.SentenceEvaluator:
    """Gets the embeddings similarity evaluator from a sample of test dataset

    Args:
        test_dataset (Dataset): Test dataset 
        truncate_dim (int): For use with Matryoshka loss training - defaults to None

    Returns:
        evaluation.SentenceEvaluator: The evaluator returned
    """
    # Sample 5000 random entries from the test_dataset
    random.seed(42)
    sample_indices = random.sample(range(len(test_dataset)), 5000)
    sampled_dataset = test_dataset.select(sample_indices)
    sentences1 = sampled_dataset["sentence1"]
    sentences2 = sampled_dataset["sentence2"]
    labels = sampled_dataset["label"]

    # Set up the evaluator with binary similarity scores
    return evaluation.BinaryClassificationEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        labels=labels,
        truncate_dim=truncate_dim,
    )
    
def convert_to_pair_dataset(sentence_label_dataset: Dataset) -> Dataset:
    """Generate pair dataset from sentence-label dataset

    Args:
        sentence_label_dataset (Dataset): sentence to class

    Returns:
        Dataset: pairs of sentences with a label indicating if they are same or different
    """
    sentences = sentence_label_dataset["text"]
    labels = sentence_label_dataset["label"]
    sentence_pair_dataset = {"sentence1":[], "sentence2":[], "label":[]}
    for (i, sent1), (j, sent2) in itertools.combinations(enumerate(sentences), 2):
        # If labels are the same, it's a similar pair (1), else dissimilar (0)
        score = 1 if labels[i] == labels[j] else 0
        sentence_pair_dataset["sentence1"].append(sent1)
        sentence_pair_dataset["sentence2"].append(sent2)
        sentence_pair_dataset["label"].append(score)  
    
    return Dataset.from_dict(sentence_pair_dataset) 

def sampled_dataset(dataset: Dataset) -> Dataset:
    """sample incoming dataset to max of 500 per label

    Args:
        dataset (Dataset): incoming dataset

    Returns:
        Dataset: sampled dataset containing 500 per label
    """
    # Initialize an empty list to hold sampled subsets
    samples = []

    # Loop through each label (assuming labels are 0, 1, and 2)
    for label in set(dataset['label']):
        # Filter rows with the current label
        label_subset = dataset.filter(lambda x, label=label : x['label'] == label)
        # Shuffle and select the first 500 samples (or fewer if less are available)
        label_sample = label_subset.shuffle(seed=42).select(range(min(500, len(label_subset))))
        # Append to samples list
        samples.append(label_sample)
    
    return concatenate_datasets(samples)   
        
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get the base sentence transformer model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name).to(device)
    
    # Get the CommentedMap of config (contains paths for data and results directories)
    config = ConfigurationMixin().load_config()
    
    # pick chunks labeled with subjects (biology, physics, history assigned to labels 0, 1, 2 respectively)
    train, test = get_train_test_lists(cfg=config)
    
    # Convert to Dataset format
    train_dataset = tuples_list_to_dataset(train)
    test_dataset = tuples_list_to_dataset(test)
    
    # Sample to max of 500 per label so that the paired dataset is having max of 1500*1499/2
    train_dataset = sampled_dataset(train_dataset)
    test_dataset = sampled_dataset(test_dataset)

    # Create the paired dataset consisting of (sentence1, sentence2, score) from the text/label dataset
    train_dataset = convert_to_pair_dataset(train_dataset)
    test_dataset = convert_to_pair_dataset(test_dataset)

    # Define a loss function
    loss = losses.CoSENTLoss(model)
    
    # Instantiate the training arguments
    use_cpu = False
    no_cuda = False
    if device == "cpu":
        use_cpu = True
        no_cuda = True

    results_dir = config["paths"]["results"]
    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"{results_dir}/subject-based-encoder",
        no_cuda=no_cuda, 
        use_cpu=use_cpu,
        max_steps=2000,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        save_strategy="steps",
        weight_decay=0.01,
        metric_for_best_model="cosine_accuracy",
        greater_is_better=True, 
        save_total_limit=1,                  # Keep only the best model checkpoint    
        report_to="none",         
    )

    # Get the evaluator to test on (take only 5000 elements of test dataset to avoid too long evaluation time)
    binary_acc_evaluator = get_evaluator(test_dataset=test_dataset)

    # Define the trainer class
    trainer = SentenceTransformerTrainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,                
        loss=loss,                 
    )

    result = binary_acc_evaluator(model)
    print(f"Before training evaluation: {result}")
    
    start_time = time.time()

    # Run the train loop
    trainer.train()

    end_time = time.time()

    result = binary_acc_evaluator(model)
    print(f"After training evaluation: {result}")
    
    # Calculate the total training time
    training_time = end_time - start_time

    # Convert to hours, minutes, seconds
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)

    print(f"Training started at: {time.ctime(start_time)}")
    print(f"Training ended at: {time.ctime(end_time)}")
    print(f"Total training time: {hours} hours, {minutes} minutes, {seconds} seconds")
