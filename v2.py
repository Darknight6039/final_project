# THIS IS ME TRYING TO GET A GOOD RESULT 
# AAARGHHHHHH 
# WILL SEE YAY 
#!/usr/bin/env python3
import multiprocessing as mp
# IMPORTANT: Set the start method to "spawn" BEFORE any other imports!
mp.set_start_method("spawn", force=True)

import os
import re
import time
import json
import random
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -------------------------------
# Simplified configuration
# -------------------------------
CONFIG = {
    "random_seed": 42,
    "train_size": 0.7,
    "val_size": 0.15,
    "test_size": 0.15,
    "max_length": 128,
    "model_name": "yiyanghkust/finbert-tone",
    "num_labels": 3,
    "batch_size": 16,
    "epochs": 4,
    "learning_rate": 2e-5,
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "data_file": "financial_phrase_bank_pt_br.csv",   # Change path as needed
    "model_dir": "./fin_sentiment_model/",
    "output_dir": "./results/"
}

# -------------------------------
# Logging configuration
# -------------------------------
def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger()

# -------------------------------
# Environment setup and reproducibility
# -------------------------------
def setup_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Device selection
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info(f"Using CPU with {os.cpu_count()} threads")
        torch.set_num_threads(os.cpu_count())
    # Download necessary NLTK resources
    for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
        nltk.download(resource, quiet=True)
    return device

# -------------------------------
# Text Preprocessing
# -------------------------------
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Additional domain-specific stopwords can be added here if needed.
        self.custom_stop = {"ltd", "inc", "corporation", "corp", "plc", "holdings"}
        self.stop_words.update(self.custom_stop)

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'(\d+(\.\d+)?)%', r' percent_\1 ', text)
        text = re.sub(r'[$€£¥](\d+(\.\d+)?)', r' currency_\1 ', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'[^\w\s%]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def remove_stopwords(self, text: str) -> str:
        words = word_tokenize(text)
        return " ".join([w for w in words if w not in self.stop_words])

    def lemmatize_text(self, text: str) -> str:
        words = word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        lemmatized = []
        for word, tag in pos_tags:
            pos = tag[0].upper()
            pos_val = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}.get(pos, wordnet.NOUN)
            lemmatized.append(self.lemmatizer.lemmatize(word, pos=pos_val))
        return " ".join(lemmatized)

    def preprocess(self, text: str) -> str:
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return self.lemmatize_text(text)

# -------------------------------
# PyTorch Dataset for Sentiment Analysis
# -------------------------------
class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -------------------------------
# Training and Evaluation Functions
# -------------------------------
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {key: batch[key].to(device) for key in batch}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device)
        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    return model

def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            inputs = {key: batch[key].to(device) for key in batch}
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(inputs['labels'].cpu().numpy())
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    model.train()  # reset to train mode
    return avg_loss, accuracy, f1

def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model saved to {output_dir}")

def load_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    logging.info(f"Loaded model from {model_dir}")
    return model, tokenizer

# -------------------------------
# Training procedure
# -------------------------------
def train_new_model(config, device, preprocessor):
    # Load and explore data
    try:
        data = pd.read_csv(config["data_file"])
    except Exception as e:
        logging.error(f"Error loading data file: {e}")
        return
    logging.info(f"Loaded data of shape {data.shape}")

    # Preprocess text
    data["processed_text"] = data["text"].apply(preprocessor.preprocess)
    sentiment_map = {"positive": 0, "negative": 1, "neutral": 2}
    data["label"] = data["y"].map(sentiment_map)

    # Split data
    temp_size = config["val_size"] + config["test_size"]
    train_df, temp_df = train_test_split(
        data, test_size=temp_size, random_state=config["random_seed"], stratify=data["label"]
    )
    val_ratio = config["val_size"] / temp_size
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio), random_state=config["random_seed"], stratify=temp_df["label"]
    )
    logging.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)

    # Create datasets
    train_dataset = FinancialSentimentDataset(
        train_df["processed_text"].tolist(), train_df["label"].tolist(),
        tokenizer, config["max_length"]
    )
    val_dataset = FinancialSentimentDataset(
        val_df["processed_text"].tolist(), val_df["label"].tolist(),
        tokenizer, config["max_length"]
    )
    
    # IMPORTANT: On Apple Silicon with MPS, using workers can cause issues.
    # If using MPS (or even CPU) and you want stability, consider setting num_workers=0.
    num_workers = 0 if device.type in ["cpu", "mps"] else 2

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type != "cpu"))
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type != "cpu"))

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], eps=config["epsilon"], weight_decay=config["weight_decay"])
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train model
    model = train_model(model, train_loader, val_loader, optimizer, scheduler, config["epochs"], device)

    # Evaluate on test set
    test_dataset = FinancialSentimentDataset(
        test_df["processed_text"].tolist(), test_df["label"].tolist(),
        tokenizer, config["max_length"]
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type != "cpu"))
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, device)
    logging.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    # Save the trained model
    save_model(model, tokenizer, config["model_dir"])

# -------------------------------
# Prediction procedure for a single text
# -------------------------------
def predict_single_text(text, config, device, preprocessor):
    model, tokenizer = load_model(config["model_dir"], device)
    processed = preprocessor.preprocess(text)
    encoding = tokenizer.encode_plus(
        processed, add_special_tokens=True, max_length=config["max_length"],
        padding='max_length', truncation=True, return_attention_mask=True,
        return_token_type_ids=True, return_tensors='pt'
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    pred_label = torch.argmax(probs).item()
    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    print("Input Text:", text)
    print("Processed Text:", processed)
    print("Predicted Sentiment:", label_map.get(pred_label, "Unknown"))
    print("Probabilities:", {label_map[i]: float(prob) for i, prob in enumerate(probs)})

# -------------------------------
# Main entry point
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simplified Financial Sentiment Analysis")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], default="train", help="Mode of operation")
    parser.add_argument("--text", type=str, help="Text for prediction mode")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logger = setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    device = setup_environment(CONFIG["random_seed"])
    preprocessor = TextPreprocessor()

    if args.mode == "train":
        train_new_model(CONFIG, device, preprocessor)
    elif args.mode == "predict":
        if not args.text:
            logging.error("Please provide text input with --text for prediction.")
        else:
            predict_single_text(args.text, CONFIG, device, preprocessor)

if __name__ == "__main__":
    main()
