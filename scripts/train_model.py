import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification,Trainer, TrainingArguments
from datasets import Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

def main():
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
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    # TRAINING
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = CrossEntropyLoss()

    model.train()
    for epoch in range(3):  # train for 3 epochs
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):   
            if batch_idx % 10 == 0:
                print(f"Loading batch {batch_idx}/{len(train_dataloader)}")
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"]
            }
            labels = batch["labels"]
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "saved_model.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()