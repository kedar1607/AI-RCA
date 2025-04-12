# This code computes the similarity between two texts using different methods
# and returns a dictionary of scores.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import Levenshtein

def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def cosine_similarity_tfidf(text1, text2):
    """
    Calculate cosine similarity using TF-IDF vectors
    Returns a score between 0 and 1
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def word_overlap_similarity(text1, text2):
    """
    Calculate similarity based on word overlap
    Returns a score between 0 and 1
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0

def levenshtein_similarity(text1, text2):
    """
    Calculate similarity using Levenshtein distance
    Returns a score between 0 and 1
    """
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0
    return 1 - (Levenshtein.distance(text1, text2) / max_len)

def get_similarity_scores(text1, text2):
    """
    Calculate similarity scores for two texts
    Returns a dictionary of scores
    """
    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    return {
        'cosine_similarity_tfidf': cosine_similarity_tfidf(processed_text1, processed_text2),
        'word_overlap': word_overlap_similarity(processed_text1, processed_text2),
        'levenshtein': levenshtein_similarity(processed_text1, processed_text2)
    }

# Example usage
# if __name__ == "__main__":
#     text1 = "The quick brown fox jumps over the lazy dog"
#     text2 = "A fast brown fox leaps over a sleepy dog"
    
#     scores = get_similarity_scores(text1, text2)
    
#     print("Similarity Scores:")
#     for method, score in scores.items():
#         print(f"{method}: {score:.4f}") 