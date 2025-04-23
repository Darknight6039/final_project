#!/usr/bin/env python
# Financial Sentiment Analysis with FinBERT (CPU‑only version)
# ============================================================
# This script fine‑tunes ProsusAI/finbert on the same train/val/test splits
# produced by `financial_sentiment_analysis_cpu.py`. Training/validation loss
# and accuracy are logged to `finbert_training.log` and saved to
# `finbert_training_logs.csv`, with curves stored to `finbert_training_metrics.png`.

import os
import time
import random
import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging -------------------------------------------------------------------
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="finbert_training.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler(); console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# ---------------------------------------------------------------------------
# Environment setup ----------------------------------------------------------
# ---------------------------------------------------------------------------

def setup_environment(seed: int = 42) -> torch.device:
    nltk.download("punkt"); nltk.download("stopwords"); nltk.download("wordnet"); nltk.download("averaged_perceptron_tagger")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(os.cpu_count())
    device = torch.device("cpu")
    logging.info("Using CPU device with %d threads", os.cpu_count())
    return device

# ---------------------------------------------------------------------------
# Text preprocessing (identical to previous script) --------------------------
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()
    return ""

def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in word_tokenize(text) if w not in stop_words])

def lemmatize_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    tag_map = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return " ".join([lemmatizer.lemmatize(tok, tag_map.get(pos[0].upper(), wordnet.NOUN)) for tok, pos in tags])

def preprocess_text(text: str) -> str:
    return lemmatize_text(remove_stopwords(clean_text(text)))

# ---------------------------------------------------------------------------
# Dataset class --------------------------------------------------------------
# ---------------------------------------------------------------------------

class FinancialSentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 128):
        self.texts = texts; self.labels = labels; self.tokenizer = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            self.texts[idx], add_special_tokens=True, max_length=self.max_length,
            padding="max_length", truncation=True, return_token_type_ids=True,
            return_attention_mask=True, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "token_type_ids": enc["token_type_ids"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ---------------------------------------------------------------------------
# Accuracy helper ------------------------------------------------------------
# ---------------------------------------------------------------------------

def flat_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return (np.argmax(preds, axis=1).flatten() == labels.flatten()).mean()

# ---------------------------------------------------------------------------
# Training loop --------------------------------------------------------------
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device, *, epochs: int = 6, lr: float = 2e-5, grad_accum: int = 4, patience: int = 2, min_delta: float = 1e-4) -> List[Dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps - int(0.1 * total_steps)))

    history = []
    best_val_loss = float("inf"); epochs_no_improve = 0
    tic = time.time()

    for epoch in range(1, epochs + 1):
        logging.info("Epoch %d/%d", epoch, epochs)
        model.train(); train_loss = 0.0
        prog_train = tqdm(train_loader, desc="  Train", leave=False)
        optimizer.zero_grad()

        for step, batch in enumerate(prog_train):
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device); tok_type = batch["token_type_ids"].to(device); labels = batch["labels"].to(device)
            loss = model(ids, attention_mask=mask, token_type_ids=tok_type, labels=labels).loss / grad_accum
            loss.backward(); train_loss += loss.item() * grad_accum
            prog_train.set_postfix(loss=f"{loss.item():.4f}")
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad()
        if len(train_loader) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # Validation -----------------------------------------------------------------
        model.eval(); val_loss = 0.0; val_acc = 0.0
        prog_val = tqdm(val_loader, desc="  Val  ", leave=False)
        for batch in prog_val:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device); tok_type = batch["token_type_ids"].to(device); labels = batch["labels"].to(device)
            with torch.no_grad():
                outputs = model(ids, attention_mask=mask, token_type_ids=tok_type, labels=labels)
            val_loss += outputs.loss.item(); val_acc += flat_accuracy(outputs.logits.detach().cpu().numpy(), labels.cpu().numpy())
            prog_val.set_postfix(val_loss=f"{outputs.loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader); avg_val_acc = val_acc / len(val_loader)
        logging.info("  train_loss=%.4f | val_loss=%.4f | val_acc=%.4f", avg_train_loss, avg_val_loss, avg_val_acc)
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_acc": avg_val_acc})

        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss; epochs_no_improve = 0
            model.save_pretrained("./best_finbert_model"); logging.info("  ** best model saved (val_loss=%.4f) **", avg_val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping after %d epochs", epoch)
                break

    minutes, seconds = divmod(time.time() - tic, 60)
    logging.info("Training finished in %dm %ds", int(minutes), int(seconds))
    return history

# ---------------------------------------------------------------------------
# Evaluation helper ----------------------------------------------------------
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, device):
    model.eval(); preds, labels = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device); tok_type = batch["token_type_ids"].to(device); y = batch["labels"].to(device)
        with torch.no_grad(): logits = model(ids, attention_mask=mask, token_type_ids=tok_type).logits.detach().cpu().numpy()
        preds.extend(np.argmax(logits, axis=1).flatten()); labels.extend(y.cpu().numpy().flatten())
    return preds, labels

# ---------------------------------------------------------------------------
# Main -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    device = setup_environment()

    # ----------------------------------------------------------------------
    # 1. Load pre‑split CSVs ----------------------------------------------
    # ----------------------------------------------------------------------
    train_path, val_path, test_path = "train.csv", "val.csv", "test.csv"
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Train/val/test CSVs not found. Run financial_sentiment_analysis_cpu.py first.")

    train_df = pd.read_csv(train_path); val_df = pd.read_csv(val_path); test_df = pd.read_csv(test_path)
    logging.info("Splits loaded: train=%d | val=%d | test=%d", len(train_df), len(val_df), len(test_df))

    # Ensure processed_text exists (it should), else build it
    if "processed_text" not in train_df.columns:
        logging.info("processed_text column missing — creating …")
        for df in (train_df, val_df, test_df):
            df["processed_text"] = df["text"].apply(preprocess_text)

    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    model.to(device)
    
    for df in (train_df, val_df, test_df):
        df["processed_text"] = df["processed_text"].fillna("").astype(str)

    def make_loader(df_split, shuffle=False):
        ds = FinancialSentimentDataset(df_split["processed_text"].tolist(), df_split["sentiment_label"].tolist(), tokenizer)
        return DataLoader(ds, batch_size=16, shuffle=shuffle, num_workers=0, pin_memory=False)

    train_loader = make_loader(train_df, shuffle=True); val_loader = make_loader(val_df); test_loader = make_loader(test_df)

    # ----------------------------------------------------------------------
    # 2. Train -------------------------------------------------------------
    # ----------------------------------------------------------------------
    history = train_model(model, train_loader, val_loader, device, epochs=6, grad_accum=4)
    history_df = pd.DataFrame(history).set_index("epoch"); history_df.to_csv("finbert_training_logs.csv")

    # ----------------------------------------------------------------------
    # 3. Plot curves -------------------------------------------------------
    # ----------------------------------------------------------------------
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["train_loss"], "b-o", label="Train"); plt.plot(history_df["val_loss"], "g-o", label="Val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_df["val_acc"], "r-o")
    plt.title("Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True)

    plt.tight_layout(); plt.savefig("finbert_training_metrics.png"); plt.close()
    logging.info("Plots saved to finbert_training_metrics.png and logs to finbert_training_logs.csv")

    # ----------------------------------------------------------------------
    # 4. Evaluate ----------------------------------------------------------
    # ----------------------------------------------------------------------
    preds, labels = evaluate_model(model, test_loader, device)
    label_names = {0: "positive", 1: "negative", 2: "neutral"}
    report = classification_report(labels, preds, target_names=list(label_names.values()))
    logging.info("\n%s", report)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(label_names.values()), yticklabels=list(label_names.values()))
    plt.title("FinBERT Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig("finbert_confusion_matrix.png"); plt.close()

    # ----------------------------------------------------------------------
    # 5. Save model --------------------------------------------------------
    # ----------------------------------------------------------------------
    out_dir = "./finbert_sentiment_model_cpu/"; os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir); tokenizer.save_pretrained(out_dir)
    logging.info("Fine‑tuned FinBERT model saved to %s", out_dir)

if __name__ == "__main__":
    main()
