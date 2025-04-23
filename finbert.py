#!/usr/bin/env python
# Financial Sentiment Analysis with FinBERT (ProsusAI/finbert)
# ===============================================================
# This script tests the Hugging Face FinBERT model on a financial sentiment
# dataset (CSV with columns: y, text, text_pt). It applies the same text
# preprocessing pipeline from your previous work and fine-tunes FinBERT,
# evaluating its loss and accuracy.
#


import os
import time
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# -------------------------------
# Setup: Download NLTK resources and fix random seed
# -------------------------------
def setup_environment(seed_value=42):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    # Use MPS if available (Apple Silicon) otherwise CPU
    #if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #    print("Using MPS device (Apple Silicon GPU).")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device with {os.cpu_count()} threads.")
    torch.set_num_threads(os.cpu_count())
    return device

# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def clean_text(text):
    """Lowercase, remove special characters and extra spaces."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def remove_stopwords(text):
    """Remove common stopwords."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def lemmatize_text(text):
    """Lemmatize text with POS tagging for improved results."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    tag_map = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    lemmatized_tokens = []
    for token, tag in pos_tags:
        pos = tag[0].upper()
        wordnet_pos = tag_map.get(pos, wordnet.NOUN)
        lemmatized_tokens.append(lemmatizer.lemmatize(token, wordnet_pos))
    return " ".join(lemmatized_tokens)

def preprocess_text(text):
    """Full preprocessing pipeline."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# -------------------------------
# Custom Dataset Class for FinBERT Fine-Tuning
# -------------------------------
class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -------------------------------
# Utility: Calculate Accuracy
# -------------------------------
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# -------------------------------
# Model Training Function
# -------------------------------

from tqdm import tqdm

def train_model(
    model,
    train_dataloader,
    val_dataloader,
    device,
    epochs=10,
    gradient_accumulation_steps=4,
    lr=2e-5,
    weight_decay=0.01,
    warmup_steps_ratio=0.1,
    patience=2,
    min_delta=1e-4,
    output_dir="./best_model"
):
    """
    Fine‑tune with:
      - AdamW(wd)
      - cosine scheduler with warmup
      - early stopping + checkpoint of best val_loss
      - tqdm progress bars
    """
    # Setup optimizer & scheduler
    total_steps = (len(train_dataloader) // gradient_accumulation_steps) * epochs
    warmup_steps = int(total_steps * warmup_steps_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps - warmup_steps,  # one cycle after warmup
        T_mult=1
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        train_loss = 0.0

        train_iter = tqdm(train_dataloader, desc="  Train", leave=False)
        for step, batch in enumerate(train_iter):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item() * gradient_accumulation_steps

            train_iter.set_postfix(loss=f"{loss.item():.4f}")

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"  → Avg Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        val_iter = tqdm(val_dataloader, desc="  Val  ", leave=False)
        for batch in val_iter:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
            loss = outputs.loss
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.to("cpu").numpy()

            val_loss += loss.item()
            val_acc += np.mean(np.argmax(logits, axis=1) == label_ids)

            val_iter.set_postfix(val_loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_acc = val_acc / len(val_dataloader)
        print(f"  → Avg Val Loss:  {avg_val_loss:.4f}")
        print(f"  → Avg Val Acc:   {avg_val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc
        })

        # Early stopping & checkpoint
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            print("  ** New best model saved **")
            model.save_pretrained(output_dir)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch} epochs.")
                break

    total_time = time.time() - start_time
    mins, secs = divmod(total_time, 60)
    print(f"\nTraining complete in {int(mins)}m {int(secs)}s.")
    return history
# -------------------------------
# Model Evaluation Function
# -------------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    
    for batch in dataloader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_token_type_ids = batch['token_type_ids'].to(device)
        b_labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=b_token_type_ids,
                            attention_mask=b_input_mask)
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())
    
    return predictions, true_labels

# -------------------------------
# Sentiment Prediction Function
# -------------------------------
def predict_sentiment(text, model, tokenizer, device):
    processed_text = preprocess_text(text)
    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)
        logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0].tolist()
    
    # FinBERT was fine-tuned for the following label mapping; adjust if needed.
    label_names = {0: 'positive', 1: 'negative', 2: 'neutral'}
    confidence = probabilities[prediction]
    if label_names[prediction] == 'positive':
        recommendation = "BUY" if confidence > 0.7 else "HOLD (leaning positive)"
    elif label_names[prediction] == 'negative':
        recommendation = "SELL" if confidence > 0.7 else "HOLD (leaning negative)"
    else:
        recommendation = "HOLD"
    
    return {
        'sentiment': label_names[prediction],
        'confidence': confidence,
        'recommendation': recommendation
    }

# -------------------------------
# MAIN EXECUTION
# -------------------------------
def main():
    # 1. Setup environment and device
    device = setup_environment(seed_value=42)
    
    # 2. Load dataset
    print("Loading dataset...")
    data_file = "financial_phrase_bank_pt_br.csv"
    df = pd.read_csv(data_file, sep=",")
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    # 3. Preprocess text (using the English 'text' column here)
    print("Preprocessing text...")
    df["processed_text"] = df["text"].apply(preprocess_text)
    
    # 4. Map sentiment labels to integers (adjust mapping to match FinBERT expectations)
    # Here we assume: positive: 0, negative: 1, neutral: 2
    sentiment_map = {"positive": 0, "negative": 1, "neutral": 2}
    df["sentiment_label"] = df["y"].map(sentiment_map)
    
    # 5. Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["sentiment_label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["sentiment_label"])
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # 6. Load FinBERT model and tokenizer from Hugging Face
    print("Loading FinBERT model and tokenizer...")
    model_name = "ProsusAI/finbert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    
    # 7. Create datasets and dataloaders
    batch_size = 16
    train_dataset = FinancialSentimentDataset(train_df["processed_text"].values,
                                              train_df["sentiment_label"].values,
                                              tokenizer,
                                              max_length=128)
    val_dataset = FinancialSentimentDataset(val_df["processed_text"].values,
                                            val_df["sentiment_label"].values,
                                            tokenizer,
                                            max_length=128)
    test_dataset = FinancialSentimentDataset(test_df["processed_text"].values,
                                             test_df["sentiment_label"].values,
                                             tokenizer,
                                             max_length=128)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)
    
    # 8. Fine‑tune FinBERT using the improved training loop
    print("Starting training with improved training loop...")
    history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        epochs=10,                       # or set to 4–10 as you wish
        gradient_accumulation_steps=4,
        lr=2e-5,
        weight_decay=0.01,
        warmup_steps_ratio=0.1,
        patience=2,
        min_delta=1e-4,
        output_dir="./best_model"
    )

    # Optional: Visualize training & validation loss and accuracy
    stats_df = pd.DataFrame(history).set_index("epoch")
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(stats_df["Training Loss"], "b-o", label="Training")
    plt.plot(stats_df["Validation Loss"], "g-o", label="Validation")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(stats_df["Validation Accuracy"], "r-o")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("finbert_training_metrics.png")
    plt.close()
    
    # 10. Evaluate on test set
    print("Evaluating on test set...")
    test_preds, test_labels = evaluate_model(model, test_dataloader, device)
    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=list(label_map.values())))
    
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(label_map.values()),
                yticklabels=list(label_map.values()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("finbert_confusion_matrix.png")
    plt.close()
    
    # 11. Test sample predictions
    example_texts = [
        "Company XYZ reports record profits for the quarter, exceeding analyst expectations by 15%.",
        "Shares plummet after earnings miss expectations with revenue declining 10% year-over-year.",
        "The market remained stable throughout the trading session with major indices showing minimal movement.",
        "Investors express concern over rising inflation rates and their impact on corporate profits.",
        "New partnership expected to boost revenue by 30% in coming years according to CEO statement."
    ]
    
    print("\nInvestment recommendations based on sample news:")
    for text in example_texts:
        result = predict_sentiment(text, model, tokenizer, device)
        print(f"\nNEWS: {text}")
        print(f"SENTIMENT: {result['sentiment'].upper()} (Confidence: {result['confidence']:.2f})")
        print(f"RECOMMENDATION: {result['recommendation']}")
    
    # 12. Save the fine-tuned model
    output_dir = "./finbert_sentiment_model/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}\nTraining and evaluation completed successfully!")

if __name__ == "__main__":
    main()
