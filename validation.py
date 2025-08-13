import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def calculate_metrics(model: nn.Module, dl: DataLoader, device: str = "cpu"):
    results = {
        "average_intraclass_similarities": [],
    }

    for images, _, (selected_pixels, target_matrix) in dl:
        images = images.to(device)
        selected_pixels = selected_pixels.to(device)
        target_matrix = target_matrix.to(device)

        with torch.no_grad():
            embeddings = model(images)

        # Calculate the average intraclass similarity
        avg_intraclass_similarities = average_intraclass_similarity(
            embeddings, selected_pixels
        )

        results["avg_intraclass_similarities"].append(avg_intraclass_similarities)

    return results


def average_intraclass_similarity(embeddings: torch.Tensor, classes: torch.Tensor):
    """
    Compute the average intraclass (cosine) similarity for the embeddings of each class.

    Args:
        embeddings (torch.Tensor): The embeddings of the samples (B, H, W, D)
        classes (torch.Tensor): The classes of the samples (B, H, W)

    Returns:
        torch.Tensor: The average intraclass similarity for each class (C,), where C is the number of unique classes and each element is between
            -1 and 1.
    """
    # Flatten the embeddings and classes to (B*H*W, D) and (B*H*W,) respectively
    embeddings_flat = embeddings.view(-1, embeddings.size(-1))  # (B*H*W, D)
    classes_flat = classes.view(-1)  # (B*H*W,)

    # Normalize the embeddings
    embeddings_flat = F.normalize(embeddings_flat, p=2, dim=1)

    # Get unique classes
    # unique_classes = torch.unique(classes_flat)
    unique_classes = torch.arange(14)

    # Initialize a tensor to store the average intraclass similarities
    avg_intraclass_similarities = torch.zeros(len(unique_classes))

    # Compute the average intraclass similarity for each class
    for i, cls in enumerate(unique_classes):
        # If the class is not present in the current batch, skip it
        if cls not in classes_flat:
            continue
        # Get embeddings for the current class
        cls_embeddings = embeddings_flat[classes_flat == cls]  # (N_cls, D)

        # Compute pairwise cosine similarity within the class
        similarity_matrix = torch.mm(cls_embeddings, cls_embeddings.T)  # (N_cls, N_cls)

        # Exclude self-similarity (diagonal elements) to avoid bias
        mask = torch.eye(
            cls_embeddings.size(0), dtype=torch.bool, device=embeddings.device
        )
        similarity_matrix = similarity_matrix[~mask].view(cls_embeddings.size(0), -1)

        # Average the similarity matrix to get the average intraclass similarity
        avg_intraclass_similarities[i] = similarity_matrix.mean()

    return avg_intraclass_similarities


def average_entropy_anomaly_similarity(embeddings: torch.Tensor, classes: torch.Tensor):
    """
    First we approximate each class embedding distribution with its mean.
    """
