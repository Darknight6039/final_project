#!/usr/bin/env python
# Financial Sentiment Analysis for Investment Decisions (CPU‑only version)
# =======================================================================
# Python script for SKEMA AI Master Project - April 2025
# This program implements a financial sentiment analysis model using BERT
# to analyze financial news and determine investment signals based on
# market sentiment. Modifications in this version:
#   • Forced CPU execution
#   • Accuracy and loss are logged to `training.log` and saved to `training_logs.csv`
#   • Train, validation and test splits are exported to `train.csv`, `val.csv`, `test.csv`
#

import os
import time
import random
import re
import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Logging configuration ------------------------------------------------------
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="training.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# ---------------------------------------------------------------------------
# Environment setup ----------------------------------------------------------
# ---------------------------------------------------------------------------

def setup_environment(seed: int = 42) -> torch.device:
    """Download NLTK resources, fix random seed and force CPU device."""
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count())
    logging.info("Using CPU device with %d threads", os.cpu_count())
    return device

# ---------------------------------------------------------------------------
# Text‑preprocessing helpers --------------------------------------------------
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return ""

def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in word_tokenize(text) if w not in stop_words])

def lemmatize_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    tag_map = {"J": nltk.corpus.wordnet.ADJ, "N": nltk.corpus.wordnet.NOUN,
               "V": nltk.corpus.wordnet.VERB, "R": nltk.corpus.wordnet.ADV}
    lemmatized = [lemmatizer.lemmatize(tok, tag_map.get(pos[0].upper(), nltk.corpus.wordnet.NOUN))
                  for tok, pos in pos_tags]
    return " ".join(lemmatized)

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = remove_stopwords(text)
    return lemmatize_text(text)

# ---------------------------------------------------------------------------
# Dataset class --------------------------------------------------------------
# ---------------------------------------------------------------------------

class FinancialSentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ---------------------------------------------------------------------------
# Training helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def flat_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return (np.argmax(preds, axis=1).flatten() == labels.flatten()).mean()

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs: int, device, grad_accum: int):
    stats = []
    tic = time.time()

    for epoch in range(epochs):
        logging.info("Epoch %d/%d", epoch + 1, epochs)
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tok_type = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(ids, attention_mask=mask, token_type_ids=tok_type, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            total_train_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if len(train_loader) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tok_type = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                outputs = model(ids, attention_mask=mask, token_type_ids=tok_type, labels=labels)
            val_loss += outputs.loss.item()
            logits = outputs.logits.detach().cpu().numpy()
            val_acc += flat_accuracy(logits, labels.to("cpu").numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        logging.info("  train_loss=%.4f | val_loss=%.4f | val_acc=%.4f", avg_train_loss, avg_val_loss, avg_val_acc)

        stats.append({
            "epoch": epoch + 1,
            "Training Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "Validation Accuracy": avg_val_acc,
        })

    minutes, seconds = divmod(time.time() - tic, 60)
    logging.info("Training completed in %dm %ds", int(minutes), int(seconds))
    return stats

# ---------------------------------------------------------------------------
# Evaluation helper ----------------------------------------------------------
# ---------------------------------------------------------------------------

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, labels_all = [], []
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tok_type = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            logits = model(ids, attention_mask=mask, token_type_ids=tok_type).logits.detach().cpu().numpy()
        preds.extend(np.argmax(logits, axis=1).flatten())
        labels_all.extend(labels.cpu().numpy().flatten())
    return preds, labels_all

# ---------------------------------------------------------------------------
# Main -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    device = setup_environment()

    # ----------------------------------------------------------------------
    # 1. Load dataset ------------------------------------------------------
    # ----------------------------------------------------------------------
    data_path = "financial_phrase_bank_pt_br.csv"
    df = pd.read_csv(data_path)
    logging.info("Dataset loaded: %s | shape=%s", data_path, df.shape)

    # ----------------------------------------------------------------------
    # 2. Pre‑process -------------------------------------------------------
    # ----------------------------------------------------------------------
    logging.info("Preprocessing text…")
    df["processed_text"] = df["text"].apply(preprocess_text)

    sentiment_map = {"positive": 0, "negative": 1, "neutral": 2}
    df["sentiment_label"] = df["y"].map(sentiment_map)

    # ----------------------------------------------------------------------
    # 3. Split & SAVE splits ----------------------------------------------
    # ----------------------------------------------------------------------
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["sentiment_label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["sentiment_label"])

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    logging.info("Splits saved to train.csv, val.csv, test.csv")

    # ----------------------------------------------------------------------
    # 4. BERT tokenizer & model -------------------------------------------
    # ----------------------------------------------------------------------
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.to(device)

    # ----------------------------------------------------------------------
    # 5. DataLoaders -------------------------------------------------------
    # ----------------------------------------------------------------------
    batch_size = 8
    grad_accum = 4

    def make_loader(split_df, shuffle=False):
        ds = FinancialSentimentDataset(split_df["processed_text"].tolist(), split_df["sentiment_label"].tolist(), tokenizer)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df)
    test_loader = make_loader(test_df)

    # ----------------------------------------------------------------------
    # 6. Optimizer & scheduler --------------------------------------------
    # ----------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = (len(train_loader) // grad_accum) * 4  # epochs=4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # ----------------------------------------------------------------------
    # 7. Train -------------------------------------------------------------
    # ----------------------------------------------------------------------
    stats = train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=4, device=device, grad_accum=grad_accum)
    stats_df = pd.DataFrame(stats).set_index("epoch")
    stats_df.to_csv("training_logs.csv")
    logging.info("Training logs saved to training_logs.csv")

    # ----------------------------------------------------------------------
    # 8. Plot training curves ---------------------------------------------
    # ----------------------------------------------------------------------
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(stats_df["Training Loss"], "b-o", label="Train")
    plt.plot(stats_df["Validation Loss"], "g-o", label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(stats_df["Validation Accuracy"], "r-o")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True)

    plt.tight_layout(); plt.savefig("training_metrics.png"); plt.close()
    logging.info("Training metric plots saved to training_metrics.png")

    # ----------------------------------------------------------------------
    # 9. Evaluate ----------------------------------------------------------
    # ----------------------------------------------------------------------
    logging.info("Evaluating on test set…")
    preds, labels = evaluate_model(model, test_loader, device)
    label_names = {0: "positive", 1: "negative", 2: "neutral"}
    report = classification_report(labels, preds, target_names=list(label_names.values()))
    logging.info("\n%s", report)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(label_names.values()), yticklabels=list(label_names.values()))
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig("confusion_matrix.png"); plt.close()

    # ----------------------------------------------------------------------
    # 10. Save model -------------------------------------------------------
    # ----------------------------------------------------------------------
    out_dir = "./financial_sentiment_model_cpu/"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logging.info("Model saved to %s", out_dir)

if __name__ == "__main__":
    main()
