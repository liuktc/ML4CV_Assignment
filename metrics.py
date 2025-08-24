# This file contains useful metrics for evaluating the model.
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def outlier_detection_roc_auc(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
    """
    Compute the ROC AUC for outlier detection.
    """
    # y_true: (N,) binary tensor where 1 indicates outlier
    # y_scores: (N,) tensor with outlier scores (higher means more likely to be outlier)
    # Normalize y_scores to [0, 1]
    assert y_true.shape == y_scores.shape
    if len(y_true.shape) > 1:
        y_true = y_true.view(-1)
        y_scores = y_scores.view(-1)
    y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-8)
    return roc_auc_score(y_true.cpu().numpy(), y_scores.cpu().numpy())
