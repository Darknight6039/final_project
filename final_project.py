# Financial Sentiment Analysis for Investment Decisions
# ====================================================
# Python script for SKEMA AI Master Project - April 2025
# This program implements a financial sentiment analysis model using BERT
# to analyze financial news and determine investment signals based on
# market sentiment. The analysis follows the project guidelines for the 
# Finance & Accounting module.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
import time
import os

# ============================================================================
# DATA UNDERSTANDING AND BUSINESS CASE
# ============================================================================
# This project addresses the financial sentiment analysis problem, which is
# critical for investment decision-making. By analyzing financial news 
# sentiments, investors can gauge market mood and potentially predict price
# movements. Our business question: "Can we accurately predict market
# sentiment from financial news to inform investment timing decisions?"

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================
# Download required NLTK resources and set random seed for reproducibility


def setup_environment():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    def set_seed(seed_value=42):
        """Set seed for reproducibility across all random number generators."""
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

    set_seed(42)

    # Select device: prefer MPS if available, else CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device with {os.cpu_count()} threads")

    # Enable Intel MKL optimization if available
    torch.set_num_threads(os.cpu_count())


# ============================================================================
# DATA PREPARATION - TEXT PREPROCESSING PIPELINE
# ============================================================================
# Implement advanced NLP preprocessing techniques including cleaning,
# lemmatization with part-of-speech tagging, and stopword removal

def clean_text(text):
    """
    Basic cleaning of texts - removes special characters, converts to lowercase
    and standardizes whitespace.
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def lemmatize_text(text):
    """
    Advanced lemmatization using WordNet lemmatizer with POS tagging.
    This improves over basic stemming by considering word context.
    """
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    
    # Get POS tags
    pos_tags = nltk.pos_tag(word_tokens)
    
    # Define mapping from POS tags to WordNet tags
    tag_map = {
        'J': nltk.corpus.wordnet.ADJ,
        'N': nltk.corpus.wordnet.NOUN,
        'V': nltk.corpus.wordnet.VERB,
        'R': nltk.corpus.wordnet.ADV
    }
    
    # Apply lemmatization with POS tag information
    lemmatized_words = []
    for word, tag in pos_tags:
        pos = tag[0].upper()  # Get first letter of POS tag
        wordnet_pos = tag_map.get(pos, nltk.corpus.wordnet.NOUN)  # Default to NOUN
        lemmatized_words.append(lemmatizer.lemmatize(word, wordnet_pos))
    
    return ' '.join(lemmatized_words)

def remove_stopwords(text):
    """Remove common stopwords that don't contribute to sentiment."""
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def preprocess_text(text):
    """Apply full preprocessing pipeline to text data."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# ============================================================================
# CUSTOM DATASET FOR BERT FINE-TUNING
# ============================================================================
# Create a PyTorch Dataset class to efficiently handle data during training

class FinancialSentimentDataset(Dataset):
    """Custom dataset for handling financial text data with BERT tokenization."""
    def __init__(self, texts, labels, tokenizer, max_length=64):
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

# Function to calculate accuracy
def flat_accuracy(preds, labels):
    """Calculate accuracy for model evaluation."""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# ============================================================================
# MODEL TRAINING
# ============================================================================
# Train the BERT model on financial sentiment data with evaluation

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, device, gradient_accumulation_steps):
    """Complete training procedure with validation and statistics."""
    training_stats = []
    total_start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print('-' * 40)
        
        total_train_loss = 0
        model.train()
        optimizer.zero_grad()  # Reset gradients at start of epoch
        
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                print(f"  Batch {step}  of  {len(train_dataloader)}")
            
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_labels = batch['labels'].to(device)
            
            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps  # Normalize loss
            loss.backward()
            
            total_train_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
        # Handle any remaining gradients at the end of epoch
        if len(train_dataloader) % gradient_accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"\n  Average training loss: {avg_train_loss:.2f}")
        
        # Validation phase
        print("\nRunning Validation...")
        model.eval()
        
        total_eval_accuracy = 0
        total_eval_loss = 0
        
        for batch in val_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_token_type_ids = batch['token_type_ids'].to(device)
            b_labels = batch['labels'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=b_token_type_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
                
            loss = outputs.loss
            logits = outputs.logits
            
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print(f"  Validation Accuracy: {avg_val_accuracy:.2f}")
        
        avg_val_loss = total_eval_loss / len(val_dataloader)
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        
        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Validation Loss': avg_val_loss,
            'Validation Accuracy': avg_val_accuracy
        })
    
    total_training_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_training_time//60:.0f}m {total_training_time%60:.0f}s")
    
    return training_stats

# ============================================================================
# MODEL EVALUATION
# ============================================================================
# Evaluate the model on the test set to measure real-world performance

def evaluate_model(model, dataloader, device):
    """Evaluate model on a given dataloader and return predictions."""
    # Put model in evaluation mode
    model.eval()
    
    # Tracking variables
    predictions = []
    true_labels = []
    
    # Evaluate data
    for batch in dataloader:
        
        # Unpack batch and move to device
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_token_type_ids = batch['token_type_ids'].to(device)
        b_labels = batch['labels'].to(device)
        
        # No gradients needed
        with torch.no_grad():
            
            # Forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_input_mask
            )
            
            logits = outputs.logits
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())
        
    return predictions, true_labels

# ============================================================================
# INVESTMENT RECOMMENDATION FUNCTION
# ============================================================================
# Function to predict sentiment for new financial statements and provide
# investment recommendations

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for new financial text."""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Tokenize and prepare for BERT
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
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    
    # Put model in evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
    
    # Get prediction class
    prediction = torch.argmax(logits, dim=1).item()
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0].tolist()
    
    # Convert to label name
    label_names = {0: 'positive', 1: 'negative', 2: 'neutral'}
    confidence = probabilities[prediction]
    
    # Investment recommendation based on sentiment
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

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    # Initialize environment
    device = setup_environment()
    
    # Load and explore data
    print("Loading and exploring data...")
    financial_data = pd.read_csv('financial_phrase_bank_pt_br.csv', sep=',')
    print(f"Dataset shape: {financial_data.shape}")
    print(financial_data.head())
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    financial_data['y'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Sentiment Distribution in Financial News')
    plt.ylabel('Number of Statements')
    plt.xlabel('Sentiment Class')
    plt.savefig('sentiment_distribution.png')
    plt.close()
    
    # Preprocess data
    print("Preprocessing text data...")
    financial_data['processed_text'] = financial_data['text'].apply(preprocess_text)
    print("Text preprocessing complete")
    print(financial_data[['text', 'processed_text', 'y']].head())
    
    # Map sentiment labels to integers
    sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    financial_data['sentiment_label'] = financial_data['y'].map(sentiment_map)
    
    # Split data
    train_data, temp_data = train_test_split(financial_data, test_size=0.3, 
                                           random_state=42, 
                                           stratify=financial_data['sentiment_label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, 
                                         random_state=42, 
                                         stratify=temp_data['sentiment_label'])
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Load pre-trained BERT tokenizer and model
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,  # Number of sentiment classes
        output_attentions=False,
        output_hidden_states=False,
    )
    
    # Move model to GPU if available
    model.to(device)
    
    # Create datasets and dataloaders
    batch_size = 8
    gradient_accumulation_steps = 4
    
    train_dataset = FinancialSentimentDataset(
        train_data['processed_text'].values,
        train_data['sentiment_label'].values,
        tokenizer
    )
    
    val_dataset = FinancialSentimentDataset(
        val_data['processed_text'].values,
        val_data['sentiment_label'].values,
        tokenizer
    )
    
    test_dataset = FinancialSentimentDataset(
        test_data['processed_text'].values,
        test_data['sentiment_label'].values,
        tokenizer
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    # Setup optimizer with gradient accumulation
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    
    # Number of training epochs
    epochs = 4
    
    # Total number of training steps
    total_steps = (len(train_dataloader) // gradient_accumulation_steps) * epochs
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Train model with gradient accumulation
    print("Starting model training...")
    training_stats = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # Visualize training results
    stats_df = pd.DataFrame(training_stats)
    stats_df = stats_df.set_index('epoch')
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(stats_df['Training Loss'], 'b-o', label='Training')
    plt.plot(stats_df['Validation Loss'], 'g-o', label='Validation')
    plt.title('Training & Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(stats_df['Validation Accuracy'], 'r-o')
    plt.title('Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_predictions, test_true_labels = evaluate_model(model, test_dataloader, device)
    
    # Convert numerical labels back to sentiment categories
    label_names = {0: 'positive', 1: 'negative', 2: 'neutral'}
    test_predictions_labels = [label_names[pred] for pred in test_predictions]
    test_true_labels_names = [label_names[label] for label in test_true_labels]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_true_labels, test_predictions, target_names=list(label_names.values())))
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(test_true_labels, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_names.values()), 
                yticklabels=list(label_names.values()))
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Test with example statements
    example_texts = [
        "Company XYZ reports record profits for the quarter, exceeding analyst expectations by 15%.",
        "Shares plummet after earnings miss expectations, with revenue declining 10% year-over-year.",
        "The market remained stable throughout the trading session, with major indices showing minimal movement.",
        "Investors express concern over rising inflation rates and potential impact on corporate profits.",
        "New partnership expected to boost revenue by 30% in coming years according to CEO statement."
    ]
    
    print("\n\nINVESTMENT RECOMMENDATIONS BASED ON SENTIMENT ANALYSIS:")
    print("="*80)
    
    for text in example_texts:
        result = predict_sentiment(text, model, tokenizer, device)
        print(f"NEWS: {text}")
        print(f"SENTIMENT: {result['sentiment'].upper()} (Confidence: {result['confidence']:.2f})")
        print(f"RECOMMENDATION: {result['recommendation']}")
        print("-"*80)
    
    # Save the model
    output_dir = './financial_sentiment_model/'
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    print("Financial sentiment analysis completed successfully!")

if __name__ == "__main__":
    main()
