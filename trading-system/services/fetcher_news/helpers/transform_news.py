# Imports
import os
import pandas as pd
from datetime import datetime, timezone
import torch
from services.fetcher_news.helpers.align_to_candle import align_to_candle_time 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import time

from services.fetcher_news.helpers.fetch_news import fetch_from_coinstats

# --- CONSTANTS & MODEL INITIALIZATION ---
HF_MODEL = os.getenv("HF_MODEL", "burakutf/finetuned-finbert-crypto")
# Move model to GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
FINBERT_LABELS = ["negative", "neutral", "positive"]

# Initialize analyzers and models
analyzer = SentimentIntensityAnalyzer()
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL).to(DEVICE)
model.eval() # Set model to evaluation mode

# -----------------------------------------------------
# 🎯 FINBERT HELPER FUNCTIONS
# -----------------------------------------------------

@torch.no_grad()
def FinBERT_sentiment_probs(text: str) -> list[float] | None:
    """
    Computes sentiment probabilities (negative, neutral, positive) using FinBERT.
    Returns probabilities as a list [neg_prob, neu_prob, pos_prob]
    """
    if not text or not isinstance(text, str):
        return None
        
    # Tokenization
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    
    # Forward pass and get logits
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Apply Softmax to get probabilities
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    
    # Round to 4 decimal places
    return [round(p, 4) for p in probs]

def FinBERT_sentiment_score(text: str) -> float | None:
    """
    Computes a single continuous sentiment score from -1 to 1 based on FinBERT probabilities.
    Score = (Positive Prob) - (Negative Prob)
    """
    probs = FinBERT_sentiment_probs(text)
    if probs is None:
        return None
        
    # probs = [negative, neutral, positive]
    neg_prob = probs[0]
    pos_prob = probs[2]
    
    # Normalized score from -1 to 1
    score = pos_prob - neg_prob
    return round(score, 4)

# -----------------------------------------------------
# 📝 VADER HELPER FUNCTION
# -----------------------------------------------------

def vader_sentiment_score(text: str) -> float:
    """Compute Vader sentiment score (compound) for the given text."""
    if not text or not isinstance(text, str):
        return 0.0
    vs = analyzer.polarity_scores(text)
    return vs['compound']

# -----------------------------------------------------
# 🚀 MAIN TRANSFORMATION FUNCTION
# -----------------------------------------------------

def transform_news(news: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw news DataFrame by selecting relevant columns, renaming them,
    aligning published_at to the candle time, and adding sentiment scores (VADER & FinBERT).
    """
    
    # Convert 'published_at' to datetime with UTC timezone
    news['published_at'] = pd.to_datetime(news['published_at'], utc=True)
    
    # 2. Candle Time Calculation and Alignment
    news['candle_time'] = align_to_candle_time(news['published_at'], interval_hours=1)
    
    # 3. VADER sentiment scoring
    news['vader_score'] = news['title'].apply(vader_sentiment_score)
    
    # 4. FinBERT scoring
    print(f"Running FinBERT inference on {len(news)} news titles using {DEVICE}...")
    
    # Score (-1 to 1)
    news['finbert_score'] = news['title'].apply(FinBERT_sentiment_score)
    
    # Label (negative, neutral, positive)
    def score_to_label(score):
        if score is None: return None
        # Normalize score from [-1, 1] to [0, 2] and round to the nearest integer
        index = int(np.round((score + 1) / 2 * (len(FINBERT_LABELS) - 1)))
        return FINBERT_LABELS[index]

    news['finbert_label'] = news['finbert_score'].apply(score_to_label)
    
    # Probabilities
    news['finbert_probs'] = news['title'].apply(FinBERT_sentiment_probs)
    
    # 5. Final Column Selection and Renaming
    final_df = news.rename(columns={
        'published_at': 'publication_time',
        'source_name': 'source', # Assuming 'source_name' column was generated in fetch_from_coinstats
    })
    
    # Define columns for the final output DataFrame
    output_cols = [
        'id', 'title', 'description', 'url', 'source', 
        'publication_time', 'candle_time', 
        'vader_score', 'finbert_score', 'finbert_label', 'finbert_probs', 'ticker'
    ]
    
    # Ensure only existing columns are selected
    valid_output_cols = [col for col in output_cols if col in final_df.columns]
    
    return final_df[valid_output_cols]

# -----------------------------------------------------
# Example Usage
# -----------------------------------------------------

if __name__ == "__main__":

    # Fetching up to 1000 posts
    df_for_analysis = fetch_from_coinstats()
    
    processed_df = transform_news(df_for_analysis)
    
    print("\n--- Processed DataFrame ---")
    print(processed_df.info())
    print("\n")
    print(processed_df.head())

