# This file contains useful metrics for evaluating the model.
import torch
from torchmetrics.classification import BinaryAveragePrecision, MulticlassJaccardIndex
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import Dataset
from typing import Literal
from tqdm.auto import tqdm


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


##############################################################################
# ANOMALY DETECTION - AUPR
##############################################################################


def compute_aupr(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
    """
    Compute the Area Under the Precision-Recall Curve (AUPR) for anomaly detection.
    Uses torchmetrics.BinaryAveragePrecision.

    Args:
        y_true (torch.Tensor): Binary ground truth labels (1 = anomaly, 0 = normal), shape (N,)
        y_scores (torch.Tensor): Continuous anomaly scores (higher = more anomalous), shape (N,)
    Returns:
        float: AUPR score
    """
    if y_true.ndim > 1:
        y_true = y_true.view(-1)
        y_scores = y_scores.view(-1)

    aupr_metric = BinaryAveragePrecision()
    return aupr_metric(y_scores, y_true).item()


##############################################################################
# SEMANTIC SEGMENTATION - mIoU
##############################################################################


def compute_mIoU(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    """
    Compute the mean Intersection over Union (mIoU) for semantic segmentation.
    Uses torchmetrics.MulticlassJaccardIndex.

    Args:
        y_true (torch.Tensor): Ground truth segmentation masks, shape (N, H, W)
        y_pred (torch.Tensor): Predicted segmentation masks, shape (N, H, W)
        num_classes (int): Number of classes
    Returns:
        float: Mean IoU score
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
    return miou_metric(y_pred.unsqueeze(0), y_true.unsqueeze(0)).item()


def compute_metrics(
    model: nn.Module,
    detector,
    test_dataset: Dataset,
    device: str,
    model_name: str = "",
    detector_name: str = "",
):
    model.eval()

    metrics = {"aupr": [], "miou": []}

    for idx in tqdm(
        range(len(test_dataset)),
        desc=f"Computing metrics for {model_name} using {detector_name}:",
    ):
        test_image, test_segmentation = test_dataset[idx]
        test_image = test_image.unsqueeze(0).to(device)

        outliers_map = detector(test_image)
        outliers_gt = (test_segmentation == 13).int()

        test_logits = model(test_image)
        predicted_segmentation = (
            torch.argmax(test_logits, dim=1).squeeze(0).detach().cpu()
        )

        aupr = compute_aupr(outliers_gt.detach().cpu(), outliers_map.detach().cpu())
        miou = compute_mIoU(
            test_segmentation.detach().cpu(),
            predicted_segmentation.detach().cpu(),
            13 + 1,
        )

        metrics["aupr"].append(aupr)
        metrics["miou"].append(miou)

    results = {}
    for key in metrics.keys():
        results[key] = {}
        results[key]["mean"] = sum(metrics[key]) / len(metrics[key])
        results[key]["std"] = torch.std(torch.tensor(metrics[key])).item()

    return results
