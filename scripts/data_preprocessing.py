import pandas as pd
import os
import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(42)

# Define emotion categories for mapping
SAFE_EMOTIONS = [
    "admiration", "approval", "caring", "contentment", "curiosity",
    "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"
]

NEEDS_ATTENTION_EMOTIONS = [
    "amusement", "confusion", "embarrassment", "fear", "nervousness",
    "realization", "remorse", "surprise"
]

CRISIS_EMOTIONS = [
    "anger", "annoyance", "disappointment", "disgust", "grief", "sadness",
    "despair"
]

# Load the GoEmotions dataset
def load_goemotions():
    dataset = load_dataset("go_emotions", split="train")
    return pd.DataFrame(dataset)

# Map original multi-label emotions into three broad categories
def map_emotions(df):
    def map_label(labels_list):
        mapped_labels = set()
        for label_id in labels_list:
            emotion = id2label[label_id]
            if emotion in SAFE_EMOTIONS:
                mapped_labels.add("safe")
            elif emotion in NEEDS_ATTENTION_EMOTIONS:
                mapped_labels.add("attention")
            elif emotion in CRISIS_EMOTIONS:
                mapped_labels.add("crisis")
        # Prioritize "crisis" if multiple categories are present
        if "crisis" in mapped_labels:
            return "crisis"
        elif "attention" in mapped_labels:
            return "attention"
        else:
            return "safe"

    df['label_mapped'] = df['labels'].apply(map_label)
    return df[['text', 'label_mapped']]

# Save train, validation, and test splits
def save_splits(df, output_dir="data/processed"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42, 
        stratify=df['label_mapped']
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['label_mapped']
    )

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Data saved to {output_dir}/")

# Map from GoEmotions ID to label name
id2label = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval',
    5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment',
    10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement',
    14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
    20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse',
    25: 'sadness', 26: 'surprise', 27: 'neutral'
}

# Main function to run preprocessing
if __name__ == "__main__":
    print("Loading GoEmotions dataset...")
    df = load_goemotions()
    print(f"Original dataset size: {len(df)}")

    print("Mapping emotions to 3 classes (safe / attention / crisis)...")
    df_mapped = map_emotions(df)
    print(df_mapped['label_mapped'].value_counts())

    print("Saving train/val/test splits...")
    save_splits(df_mapped)
    print("Preprocessing completed!")

