import torch
from torch.nn import CrossEntropyLoss

def evaluate(model, val_dataloader, device="cpu"):
    model.eval()
    model.to(device)
    loss_fn = CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            labels = batch["label_mapped"].to(device)

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_dataloader)
    accuracy = correct / total

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    return avg_loss, accuracy

# Example usage:
# evaluate(model, val_dataloader, device=device)
