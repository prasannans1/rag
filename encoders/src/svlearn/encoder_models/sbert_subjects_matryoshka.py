#  ------------------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------------------

import time
import torch
from sentence_transformers import (SentenceTransformer, 
                                   SentenceTransformerTrainer, 
                                   SentenceTransformerTrainingArguments, 
                                   losses)
from sentence_transformers.evaluation import SequentialEvaluator
from svlearn.config.configuration import ConfigurationMixin
from svlearn.encoder_models.sbert_subjects_full_ft import convert_to_pair_dataset, get_evaluator, sampled_dataset
from svlearn.util.hf_text_util import get_train_test_lists, tuples_list_to_dataset
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get the base sentence transformer model
    model_name = "microsoft/mpnet-base"
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
    inner_train_loss = losses.CoSENTLoss(model)
    
    # Add matryoshka loss
    matryoshka_dims = [768, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    train_loss = losses.MatryoshkaLoss(model, 
                                       loss=inner_train_loss, 
                                       matryoshka_dims=matryoshka_dims)
    
    # Instantiate the training arguments
    use_cpu = False
    no_cuda = False
    if device == "cpu":
        use_cpu = True
        no_cuda = True

    results_dir = config["paths"]["results"]
    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"{results_dir}/subject-based-encoder-matryoshka",
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
    evaluators = []
    for dim in matryoshka_dims:
        evaluators.append(
            get_evaluator(test_dataset=test_dataset, truncate_dim=dim)
        )
    test_evaluator = SequentialEvaluator(evaluators)
    
    # Define the trainer class
    trainer = SentenceTransformerTrainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,                
        loss=train_loss,                 
    )

    result = test_evaluator(model)
    print(f"Before training evaluation: {result}")
    
    start_time = time.time()

    # Run the train loop
    trainer.train()

    end_time = time.time()

    result = test_evaluator(model)
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
