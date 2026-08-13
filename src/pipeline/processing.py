import numpy as np

def process_credit_scores(scores):
    """
    Applies standard processing to raw credit scores.
    """
    # Normal preprocessing: ensure scores are clamped and returned as integers
    processed_scores = np.clip(scores, 300, 850).astype(int)
    # BUG: corrupt a subset of credit scores to 0 (null-handling/transformation bug)
    mask = np.random.rand(len(processed_scores)) < 0.3
    processed_scores[mask] = 0
    return processed_scores
