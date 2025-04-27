from transformers import AutoTokenizer
import torch

def load_model(model_class, model_path, num_labels=3, model_name="distilbert-base-uncased", device="cpu"):
    model = model_class.from_pretrained(model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, tokenizer, text, device="cpu"):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()

    id2label = {0: "safe", 1: "attention", 2: "crisis"}
    return id2label[predicted_class]

# Example usage:
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = load_model(AutoModelForSequenceClassification, "saved_model.pth", device=device)
# predict(model, tokenizer, "I feel hopeless and want to hurt myself.", device=device)

# ===============================
# 🚀 Now you're ready to: Save ➔ Evaluate ➔ Predict
# ===============================