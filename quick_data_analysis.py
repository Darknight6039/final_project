#!/usr/bin/env python
"""Quick exploratory analysis for the Financial Phrase Bank dataset.

Run with:
    python quick_data_analysis.py          # uses default CSV in same folder
    python quick_data_analysis.py my.csv   # custom CSV path

Outputs are written to a folder called "data analysis" (space preserved):
  • basic_info.txt          – shape, dtypes, null counts (after dropping `text_pt`)
  • class_distribution.png  – bar chart of sentiment labels
  • text_length_hist.png    – histogram of token counts
  • top_words.csv           – 100 most frequent cleaned words

The Portuguese column `text_pt` is ignored because it is not used in the
English‑language sentiment model.
"""

import sys
import os
import re
import collections
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure minimal NLTK downloads (quiet)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lower‑case, strip punctuation and condense whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main(csv_path: str = "financial_phrase_bank_pt_br.csv") -> None:
    # Resolve path ----------------------------------------------------------------
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Provide a valid CSV path.")
        sys.exit(1)

    # Create output directory ------------------------------------------------------
    out_dir = Path("data analysis")
    out_dir.mkdir(exist_ok=True)

    # Load data --------------------------------------------------------------------
    df = pd.read_csv(csv_path)

    # Drop Portuguese column if present -------------------------------------------
    if "text_pt" in df.columns:
        df = df.drop(columns=["text_pt"])

    # Save basic info --------------------------------------------------------------
    info_lines = [
        f"File: {csv_path.name}",
        f"Shape: {df.shape}",
        "\nColumn types:\n" + df.dtypes.to_string(),
        "\n\nNull counts:\n" + df.isnull().sum().to_string(),
    ]
    (out_dir / "basic_info.txt").write_text("\n".join(info_lines), encoding="utf-8")

    # Class distribution -----------------------------------------------------------
    if "y" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x="y", data=df, palette=["green", "red", "blue"])
        plt.title("Sentiment Class Distribution")
        plt.xlabel("Sentiment label"); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "class_distribution.png")
        plt.close()

    # Histogram of text length -----------------------------------------------------
    if "text" in df.columns:
        lengths = df["text"].astype(str).apply(lambda t: len(word_tokenize(t)))
        plt.figure(figsize=(8, 5))
        sns.histplot(lengths, bins=40, kde=False)
        plt.title("Text Length Distribution (tokens)")
        plt.xlabel("Tokens per row"); plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / "text_length_hist.png")
        plt.close()

        # Top 100 words ----------------------------------------------------------
        stop_words = set(stopwords.words("english"))
        all_tokens = []
        for t in df["text"].astype(str):
            all_tokens.extend([w for w in word_tokenize(clean_text(t)) if w and w not in stop_words])
        top_words = collections.Counter(all_tokens).most_common(100)
        pd.DataFrame(top_words, columns=["token", "count"]).to_csv(out_dir / "top_words.csv", index=False)

    print(f"Data analysis files saved to '{out_dir}/'")


if __name__ == "__main__":
    # Allow optional path argument; default if absent
    main(sys.argv[1] if len(sys.argv) > 1 else "financial_phrase_bank_pt_br.csv")
