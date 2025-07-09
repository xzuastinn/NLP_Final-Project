# preprocessing.py
# ------------------------------------------
# This module handles the data cleaning,
# sentiment mapping, and text preprocessing steps
# for the TripAdvisor sentiment analysis project.
# ------------------------------------------

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Clean the text (lowercase, remove punctuation, stopwords)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered_tokens)

# Map numerical TripAdvisor ratings (1–5) to sentiment labels
def map_rating_to_sentiment(score):
    """
    Convert numeric score to sentiment label:
    Positive: 4–5, Neutral: 3, Negative: 1–2
    """
    if score >= 4:
        return 'positive'
    elif score == 3:
        return 'neutral'
    else:
        return 'negative'

# Apply cleaning and TF-IDF vectorization
# Add a label column but retain original rating column
def preprocess_for_model(df):
    """
    Assumes df has 'review' and 'rating' columns.
    Applies cleaning, maps sentiment to a new 'label' column,
    and returns train/test sets.
    """
    df['label'] = df['Rating'].apply(map_rating_to_sentiment)
    df['cleaned'] = df['Review'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned'])
    y = df['label']

    return train_test_split(X, y, test_size=0.2, random_state=42), tfidf