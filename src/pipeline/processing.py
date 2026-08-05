import numpy as np

def process_credit_scores(scores):
    """
    Applies standard processing to raw credit scores.
    """
    # Normal preprocessing: ensure scores are clamped and returned as integers
    processed_scores = np.clip(scores, 300, 850).astype(int)
    return processed_scores
