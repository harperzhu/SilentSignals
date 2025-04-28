import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification,Trainer, TrainingArguments
from datasets import Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

# Load the processed train/val csv files
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')

label2id = {
    "safe": 0,
    "attention": 1,
    "crisis": 2
}

train_df['label_mapped'] = train_df['label_mapped'].map(label2id)
val_df['label_mapped'] = val_df['label_mapped'].map(label2id)

# Convert pandas to Huggingface Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

model_name = "distilbert-base-uncased"
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3 
)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

train_dataset = train_dataset.rename_column("label_mapped", "labels")
val_dataset = val_dataset.rename_column("label_mapped", "labels")

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4,num_workers=4)

# TRAINING
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss()

training_args = TrainingArguments(
    output_dir="./saved_silentsignals",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

trainer.save_model("./saved_silentsignals")

