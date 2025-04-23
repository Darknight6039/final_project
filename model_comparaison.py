# model_comparaison.py
# ==========================================
# This script evaluates and compares the performance of base and fine-tuned
# BERT models on a financial sentiment analysis task.
# Author: SKEMA AI Master Student - April 2025

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging
import re
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_comparison.log"),
        logging.StreamHandler()
    ]
)

class FinancialSentimentDataset(Dataset):
    """Dataset for financial sentiment analysis."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def evaluate_model(model, dataloader, device):
    """Evaluate model performance on the given dataloader."""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = (np.array(predictions) == np.array(true_labels)).mean()
    return accuracy, predictions, true_labels

def load_base_model(model_name, num_labels=3, device="cpu"):
    """Load a base pre-trained model from Hugging Face."""
    logging.info(f"Loading base model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.to(device)
    return model, tokenizer

def load_fine_tuned_model(model_path, device="cpu"):
    """Load a locally fine-tuned model."""
    logging.info(f"Loading fine-tuned model from: {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"Model path not found: {model_path}")
        return None, None
    
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.to(device)
    return model, tokenizer

def plot_confusion_matrix(true_labels, predictions, model_name, class_names=["positive", "negative", "neutral"]):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_confusion_matrix.png")
    plt.close()
    logging.info(f"Confusion matrix plot saved for {model_name}")
    
def plot_distribution(predictions, model_name, class_names=["positive", "negative", "neutral"]):
    """Plot and save class distribution."""
    plt.figure(figsize=(5, 4))
    # Count occurrences of each class
    unique, counts = np.unique(predictions, return_counts=True)
    counts_dict = {i: 0 for i in range(len(class_names))}
    for i, count in zip(unique, counts):
        counts_dict[i] = count
    
    plt.bar(range(len(class_names)), [counts_dict[i] for i in range(len(class_names))], color=['#66c2a5', '#fc8d62', '#8da0cb'])
    plt.title(f"Predicted Class Distribution: {model_name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.xticks(range(len(class_names)), class_names)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_predicted_distribution.png")
    plt.close()
    logging.info(f"Predicted class distribution plot saved for {model_name}")

def create_comparison_chart(base_accuracies, ft_accuracies, model_names, output_path="model_comparison.png"):
    """Create a bar chart comparing base and fine-tuned model accuracies."""
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, base_accuracies, width, label='Base Model')
    plt.bar(x + width/2, ft_accuracies, width, label='Fine-tuned Model')
    
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(x, model_names)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on bars
    for i, v in enumerate(base_accuracies):
        plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center')
    
    for i, v in enumerate(ft_accuracies):
        if v > 0:  # Only add text if we have a value
            plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Comparison chart saved to {output_path}")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Define model names and paths
    bert_base_name = "bert-base-uncased"
    finbert_base_name = "ProsusAI/finbert"
    
    # Set paths to fine-tuned models
    bert_ft_path = "./financial_sentiment_model_cpu/"
    finbert_ft_path = "./finbert_sentiment_model_cpu/"
    
    # Load test data
    test_path = "test.csv"
    if not os.path.exists(test_path):
        logging.error(f"Test data not found: {test_path}")
        return
    
    test_df = pd.read_csv(test_path)
    logging.info(f"Loaded test data: {len(test_df)} examples")
    
    # Check if processed_text column exists
    if "processed_text" not in test_df.columns:
        logging.error("processed_text column not found in test data")
        return
    
    # Initialize results dictionary
    results = {
        "Models": [],
        "Accuracy": [],
        "Type": []
    }
    
    batch_size = 16
    test_texts = test_df["processed_text"].tolist()
    test_labels = test_df["sentiment_label"].tolist()
    class_names = ["positive", "negative", "neutral"]
    
    # Evaluate BERT base model
    bert_base_model, bert_base_tokenizer = load_base_model(bert_base_name, device=device)
    bert_base_dataset = FinancialSentimentDataset(
        test_texts,
        test_labels,
        bert_base_tokenizer
    )
    bert_base_loader = DataLoader(bert_base_dataset, batch_size=batch_size, shuffle=False)
    bert_base_acc, bert_base_preds, bert_base_true = evaluate_model(bert_base_model, bert_base_loader, device)
    logging.info(f"BERT base model accuracy: {bert_base_acc:.4f}")
    
    # Add to results
    results["Models"].append("BERT")
    results["Accuracy"].append(bert_base_acc)
    results["Type"].append("Base")
    
    # Plot BERT base results
    plot_confusion_matrix(bert_base_true, bert_base_preds, "BERT base", class_names)
    plot_distribution(bert_base_preds, "BERT", class_names)
    
    # Write BERT base classification report to file
    bert_report = classification_report(bert_base_true, bert_base_preds, target_names=["positive", "negative", "neutral"])
    with open("model_classification_report.txt", "w") as f:
        f.write("BERT Base Classification Report:\n")
        f.write(bert_report)
        f.write("\n\n")
    
    # Evaluate FinBERT base model
    finbert_base_model, finbert_base_tokenizer = load_base_model(finbert_base_name, device=device)
    finbert_base_dataset = FinancialSentimentDataset(
        test_texts,
        test_labels,
        finbert_base_tokenizer
    )
    finbert_base_loader = DataLoader(finbert_base_dataset, batch_size=batch_size, shuffle=False)
    finbert_base_acc, finbert_base_preds, finbert_base_true = evaluate_model(finbert_base_model, finbert_base_loader, device)
    logging.info(f"FinBERT base model accuracy: {finbert_base_acc:.4f}")
    
    # Add to results
    results["Models"].append("FinBERT")
    results["Accuracy"].append(finbert_base_acc)
    results["Type"].append("Base")
    
    # Plot FinBERT base results
    plot_confusion_matrix(finbert_base_true, finbert_base_preds, "FinBERT base", class_names)
    plot_distribution(finbert_base_preds, "FinBERT", class_names)
    
    # Write FinBERT base classification report to file
    finbert_report = classification_report(finbert_base_true, finbert_base_preds, target_names=["positive", "negative", "neutral"])
    with open("model_classification_report.txt", "a") as f:
        f.write("FinBERT Base Classification Report:\n")
        f.write(finbert_report)
        f.write("\n")
    
    # Print classification reports
    print("\nBERT Base Classification Report:")
    print(classification_report(bert_base_true, bert_base_preds, target_names=class_names))
    
    print("\nFinBERT Base Classification Report:")
    print(classification_report(finbert_base_true, finbert_base_preds, target_names=class_names))
    
    # Evaluate fine-tuned models if available
    bert_ft_acc = finbert_ft_acc = None
    
    # Evaluate fine-tuned BERT model if available
    if os.path.exists(bert_ft_path):
        bert_ft_model, bert_ft_tokenizer = load_fine_tuned_model(bert_ft_path, device)
        bert_ft_dataset = FinancialSentimentDataset(
            test_texts,
            test_labels,
            bert_ft_tokenizer
        )
        bert_ft_loader = DataLoader(bert_ft_dataset, batch_size=batch_size, shuffle=False)
        bert_ft_acc, bert_ft_preds, bert_ft_true = evaluate_model(bert_ft_model, bert_ft_loader, device)
        logging.info(f"BERT fine-tuned model accuracy: {bert_ft_acc:.4f}")
        
        # Add to results
        results["Models"].append("BERT")
        results["Accuracy"].append(bert_ft_acc)
        results["Type"].append("Fine-tuned")
        
        # Plot BERT fine-tuned results
        plot_confusion_matrix(bert_ft_true, bert_ft_preds, "BERT fine-tuned", class_names)
        plot_distribution(bert_ft_preds, "BERT fine-tuned", class_names)
        
        # Calculate improvement
        bert_improvement = bert_ft_acc - bert_base_acc
        bert_improvement_percent = (bert_improvement / bert_base_acc) * 100
        logging.info(f"BERT improvement: {bert_improvement:.4f} ({bert_improvement_percent:.2f}%)")
        
        print("\nBERT Fine-tuned Classification Report:")
        print(classification_report(bert_ft_true, bert_ft_preds, target_names=class_names))
    else:
        logging.warning("Fine-tuned BERT model not found. Skipping evaluation.")
    
    # Evaluate fine-tuned FinBERT model if available
    if os.path.exists(finbert_ft_path):
        finbert_ft_model, finbert_ft_tokenizer = load_fine_tuned_model(finbert_ft_path, device)
        finbert_ft_dataset = FinancialSentimentDataset(
            test_texts,
            test_labels,
            finbert_ft_tokenizer
        )
        finbert_ft_loader = DataLoader(finbert_ft_dataset, batch_size=batch_size, shuffle=False)
        finbert_ft_acc, finbert_ft_preds, finbert_ft_true = evaluate_model(finbert_ft_model, finbert_ft_loader, device)
        logging.info(f"FinBERT fine-tuned model accuracy: {finbert_ft_acc:.4f}")
        
        # Add to results
        results["Models"].append("FinBERT")
        results["Accuracy"].append(finbert_ft_acc)
        results["Type"].append("Fine-tuned")
        
        # Plot FinBERT fine-tuned results
        plot_confusion_matrix(finbert_ft_true, finbert_ft_preds, "FinBERT fine-tuned", class_names)
        plot_distribution(finbert_ft_preds, "FinBERT fine-tuned", class_names)
        
        # Calculate improvement
        finbert_improvement = finbert_ft_acc - finbert_base_acc
        finbert_improvement_percent = (finbert_improvement / finbert_base_acc) * 100
        logging.info(f"FinBERT improvement: {finbert_improvement:.4f} ({finbert_improvement_percent:.2f}%)")
        
        print("\nFinBERT Fine-tuned Classification Report:")
        print(classification_report(finbert_ft_true, finbert_ft_preds, target_names=class_names))
    else:
        logging.warning("Fine-tuned FinBERT model not found. Skipping evaluation.")
    
    # Create comparison chart
    model_names = ["BERT", "FinBERT"]
    base_accs = [bert_base_acc, finbert_base_acc]
    ft_accs = [bert_ft_acc if bert_ft_acc is not None else 0, 
               finbert_ft_acc if finbert_ft_acc is not None else 0]
    
    create_comparison_chart(base_accs, ft_accs, model_names)
    
    # Create results table
    results_df = pd.DataFrame(results)
    results_pivot = results_df.pivot(index="Models", columns="Type", values="Accuracy")
    
    # Calculate improvement
    if "Fine-tuned" in results_pivot.columns:
        results_pivot["Improvement"] = results_pivot["Fine-tuned"] - results_pivot["Base"]
        results_pivot["Improvement (%)"] = (results_pivot["Improvement"] / results_pivot["Base"]) * 100
    
    logging.info("\n=== MODEL COMPARISON RESULTS ===\n")
    logging.info(f"\n{results_pivot}")
    
    # Save results to CSV
    results_pivot.to_csv("model_comparison_results.csv")
    logging.info("Results saved to model_comparison_results.csv")

if __name__ == "__main__":
    main()
