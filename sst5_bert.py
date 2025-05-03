import nltk
import os
from nltk.tree import Tree
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import torch
import time
from sklearn.metrics import classification_report

# Set NLTK data path
nltk_data_path = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download NLTK data
try:
    nltk.download('punkt', download_dir=nltk_data_path)
except Exception as e:
    print(f"Error downloading NLTK punkt: {e}")
    raise

def parse_sst5_tree(line):
    """Parse a single SST-5 tree line into phrases and labels."""
    try:
        if not line.strip():
            return []
        tree = Tree.fromstring(line)
        phrases = []
        
        def extract_phrases(subtree):
            if isinstance(subtree, Tree):
                try:
                    label = int(subtree.label())
                    phrase = " ".join(subtree.leaves())
                    if len(subtree.leaves()) >= 2:
                        phrases.append({"text": phrase, "label": label})
                    for child in subtree:
                        extract_phrases(child)
                except ValueError:
                    pass
        
        extract_phrases(tree)
        return phrases
    except Exception as e:
        print(f"Error parsing tree: {line[:50]}...: {e}")
        return []

def load_sst5_data(file_path):
    """Load phrases and labels from SST-5 tree file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        all_phrases = []
        for line in lines:
            phrases = parse_sst5_tree(line.strip())
            all_phrases.extend(phrases)
        if not all_phrases:
            raise ValueError("No valid phrases extracted from dataset")
        print(f"Loaded {len(all_phrases)} phrases from {file_path}")
        return all_phrases
    except Exception as e:
        print(f"Error loading SST-5 data: {e}")
        raise

def preprocess_data():
    """Load and convert SST-5 data to Dataset format with reduced size."""
    train_data = load_sst5_data("trainDevTestTrees_PTB/trees/train.txt")[:10000]
    test_data = load_sst5_data("trainDevTestTrees_PTB/trees/test.txt")[:2000]
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    return train_dataset, test_dataset

def tokenize_data(data, tokenizer):
    """Tokenize dataset for BERT."""
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    return data.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    """Compute classification metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(labels, predictions, 
                                  target_names=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
                                  output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"]
    }

def train_model():
    """Fine-tune DistilBERT, save model, and print test accuracy."""
    start_time = time.time()
    
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    train_data, test_data = preprocess_data()
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    
    train_data = tokenize_data(train_data, tokenizer)
    test_data = tokenize_data(test_data, tokenizer)
    
    train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    test_accuracy = eval_results["eval_accuracy"]
    print(f"Test set accuracy: {test_accuracy:.4f}")
    
    # Save model and tokenizer
    model.save_pretrained("sst5_distilbert")
    tokenizer.save_pretrained("sst5_distilbert")
    print("Model and tokenizer saved to sst5_distilbert/")
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return tokenizer, model

def predict_sentiment(tokenizer, model, text):
    """Predict sentiment for a given text and measure inference time."""
    start_time = time.time()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model = model.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    label_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    
    inference_time = time.time() - start_time
    return label_map[prediction], inference_time

if __name__ == "__main__":
    tokenizer, model = train_model()
    
    test_phrases = ["This movie is not good", "Absolutely fantastic film", "It was okay"]
    total_inference_time = 0
    for phrase in test_phrases:
        sentiment, inference_time = predict_sentiment(tokenizer, model, phrase)
        total_inference_time += inference_time
        print(f"Phrase: '{phrase}' -> Sentiment: {sentiment}, Inference time: {inference_time:.4f} seconds")
    
    print(f"Total inference time for {len(test_phrases)} phrases: {total_inference_time:.4f} seconds")
    print(f"Average inference time per phrase: {total_inference_time/len(test_phrases):.4f} seconds")