import os
import json
from sklearn.base import defaultdict
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pad
import random

from tqdm.auto import tqdm


def sample_pixels_per_class(X, labels, num_samples_per_class):
    """
    Sample a fixed number of pixels per class from the segmentation labels.

    Args:
        X (torch.Tensor): Input tensor of shape (N, D).
        labels (torch.Tensor): Segmentation labels of shape (N, ).
        num_samples_per_class (int): Number of pixels to sample per class.
    Returns:
        torch.Tensor: Tensor containing sampled pixels.
    """
    classes = torch.unique(labels)
    sampled_pixels = []
    sampled_labels = []

    for c in classes:
        if c == 255:  # Skip ignore class
            continue

        mask = labels == c
        pixels = X[mask]  # shape: (num_pixels_c, D)
        if pixels.size(0) > num_samples_per_class:
            indices = torch.randperm(pixels.size(0))[:num_samples_per_class]
            pixels = pixels[indices]

        sampled_pixels.append(pixels)
        sampled_labels.extend([c] * pixels.size(0))

    return torch.cat(sampled_pixels, dim=0), torch.tensor(
        sampled_labels, dtype=torch.long
    )
