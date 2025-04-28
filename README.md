# SilentSignals

Lightweight NLP system for early crisis signal detection.

SilentSignals aims to detect early signs of psychological distress (such as crisis or needs for attention) from short text inputs.  
Built with Huggingface Transformers and PyTorch, optimized for lightweight deployment.

---

## Project Structure

- `data/` — Raw and processed datasets
- `scripts/` — Preprocessing and training scripts
- `README.md` — Project overview and setup instructions
- `requirements.txt` — Environment dependencies

---

## How to Run

1. Create environment:

   ```bash
   conda create -n silentsignals python=3.10
   conda activate silentsignals
   pip install -r requirements.txt
   ```

2. Train model:

   ```bash
   python scripts/train_model.py
   ```

3. Model Details
   Backbone: distilbert-base-uncased

Classes: safe / needs_attention / crisis

Max sequence length: 128 tokens

Optimizer: AdamW

Loss Function: CrossEntropyLoss

Training Epochs: 3 (default)

## Tech stack map

RAW TEXT DATA
↓
Data Preprocessing - Label Mapping (safe / attention / crisis) - Cleaning / Formatting
↓
Tokenization - DistilBERT Tokenizer - Outputs: input_ids, attention_mask
↓
Deep Learning Model - Backbone: DistilBERT (Transformer Encoder) - Head: Classification layer (3 classes)
↓
Training - Loss: CrossEntropyLoss - Optimizer: AdamW
↓
Model Output - Class prediction: safe / attention / crisis
