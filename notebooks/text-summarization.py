# Databricks notebook source

import os

#change the working directory to a level above the current working directory
os.chdir("..")

# COMMAND ----------

# MAGIC %md ##Install all libraries from requirements.txt

# COMMAND ----------

# MAGIC %pip install -r requirements.txt


# COMMAND ----------
from transformers import pipeline, set_seed
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset, load_metric

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
import torch

nltk.download("punkt")

# COMMAND ----------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# COMMAND ----------


model_ckpt = "google/pegasus-cnn_dailymail"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

# COMMAND ----------

# MAGIC %sh wget https://github.com/entbappy/Branching-tutorial/raw/master/summarizer-data.zip 

# COMMAND ----------

# MAGIC %sh unzip summarizer-data.zip

# COMMAND ----------
dataset_samsum = load_from_disk('samsum_dataset')
dataset_samsum

# COMMAND ----------

split_lengths = [len(dataset_samsum[split])for split in dataset_samsum]

print(f"Split lengths: {split_lengths}")
print(f"Features: {dataset_samsum['train'].column_names}")
print("\nDialogue:")

print(dataset_samsum["test"][1]["dialogue"])

print("\nSummary:")

print(dataset_samsum["test"][1]["summary"])

# COMMAND ----------
# MAGIC %md ## Preprocessing

# COMMAND ----------
def convert_examples_to_features(example_batch):
    """
    Converts an example batch into input and target encodings.
    
    Args:
        example_batch: A dictionary containing input and target sequences.
        
    Returns:
        A dictionary with input and target encodings.
    """
    
    # Encode the dialogue sequence using the tokenizer
    input_encodings = tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
    
    # Encode the summary sequence as the target using the target tokenizer
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['summary'], max_length=128, truncation=True)
        
    # Return a dictionary with input and target encodings
    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

# COMMAND ----------

dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched = True)

# COMMAND ----------

dataset_samsum_pt["train"]

# COMMAND ----------
# MAGIC %md ## Training

# COMMAND ----------
from transformers import DataCollatorForSeq2Seq

# Create a data collator for sequence-to-sequence models
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)


# COMMAND ----------
from transformers import TrainingArguments, Trainer

# Define the training arguments for the Trainer
trainer_args = TrainingArguments(
    output_dir='pegasus-samsum', num_train_epochs=1, warmup_steps=500,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    weight_decay=0.01, logging_steps=10,
    evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
    gradient_accumulation_steps=16
)

#COMMAND ----------
# Define the Trainer object
trainer = Trainer(model = model_pegasus, args = trainer_args,
                  tokenizer = tokenizer, data_collator = seq2seq_data_collator,
                  train_dataset = dataset_samsum_pt["train"],
                  eval_dataset = dataset_samsum_pt["validation"])

trainer.train()

# COMMAND ----------
#evaluating the model
def generate_data_batches(list_of_elements, batch_size):
    '''method to split dataset into smaller batches that we can process simultaneously.
    Yield successive batch_size chunks from list_of_elements.'''
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i:i + batch_size]
        
        
        