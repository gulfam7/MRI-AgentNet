import torch
from bert_score import score


def compute_bert_score(reference_texts, candidate_texts, model_type="microsoft/deberta-large-mnli"):
    """
    Computes BERTScore similarity between reference texts and candidate texts.

    Parameters:
    - reference_texts (list): Ground truth or high-confidence responses.
    - candidate_texts (list): Generated responses to compare.
    - model_type (str): Pre-trained BERT model for scoring.

    Returns:
    - List of similarity scores (range 0-1) for each text pair.
    """

    if not reference_texts or not candidate_texts:
        print("⚠️ Warning: Empty inputs provided for BERTScore.")
        return [0] * len(candidate_texts)


    P, R, F1 = score(candidate_texts, reference_texts, model_type=model_type, lang="en", verbose=True)


    similarity_scores = F1.tolist()

    return similarity_scores
