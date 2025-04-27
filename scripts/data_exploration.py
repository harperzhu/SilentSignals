from datasets import load_dataset
import pandas as pd

# Load GoEmotions
dataset = load_dataset("go_emotions", split="train")

print(dataset[0])
print(dataset[1])
print(dataset[2])
