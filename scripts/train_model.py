import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

# Load the processed train/val csv files
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')

model_name = "distilbert-base-uncased"
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3 
)

# Convert pandas to Huggingface Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
