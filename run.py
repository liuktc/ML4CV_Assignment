import os
import json

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from plot import color, semantic_embeddings_plot
from dataset import StreetHazardDataset, PadToMultipleOf16, StreetHazardDatasetTriplet
from _model import DINOv2_SemanticSegmentation
from loss import nt_xent_loss

from model_new import DinoMetricLearning, DinoUpsampling
from loss import triplet_loss, MagnetLoss, NT_Xent_Loss, Proxy_Anchor
from train import train_metric_learning


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLAB = False
KAGGLE = False
NUM_CLASSES = 13
SEED = 42

# Seed everything
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


ORIGINAL_IMAGE_SIZE = (720, 1280)
IMAGE_SIZE = PadToMultipleOf16().convert_dims(ORIGINAL_IMAGE_SIZE)
BATCH_SIZE = 1
POSITIVE_PAIRS = True
PIXEL_PER_CLASS = 3

EPOCHS = 40
LEARNING_RATE = 1e-4
EMBEDDING_SIZE = 512

# File paths
# annotations_train_file = "/home/federico/.cache/kagglehub/datasets/lucadome/streethazards-train/versions/1/train/train.odgt"
# annotation_val_file = "/home/federico/.cache/kagglehub/datasets/lucadome/streethazards-train/versions/1/train/validation.odgt"
# annotation_test_file = "/home/federico/Downloads/streethazards_test/test/test.odgt"
# img_dir = "/home/federico/.cache/kagglehub/datasets/lucadome/streethazards-train/versions/1/train/"
# img_dir_test = "/home/federico/Downloads/streethazards_test/test/"
annotations_train_file = "./data/train/train.odgt"
annotation_val_file = "./data/train/validation.odgt"
annotation_test_file = "./data/test/test.odgt"
img_dir = "./data/train/"
img_dir_test = "./data/test/"

image_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ]
)

target_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    ]
)


# Crea il dataset
dataset_train = StreetHazardDataset(
    annotations_train_file,
    img_dir,
    image_transform=image_transform,
    target_transform=target_transform,
    positive_pairs=True,
    pixel_per_class=PIXEL_PER_CLASS,
)

dataset_val = StreetHazardDataset(
    annotation_val_file,
    img_dir,
    image_transform=image_transform,
    target_transform=target_transform,
    positive_pairs=True,
    pixel_per_class=PIXEL_PER_CLASS,
)

dataset_test = StreetHazardDataset(
    annotation_test_file,
    img_dir_test,
    image_transform=image_transform,
    target_transform=target_transform,
    positive_pairs=True,
    pixel_per_class=PIXEL_PER_CLASS,
)

dl_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
dl_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
# Sample a small random subset of training data for metrics
dl_train_small = DataLoader(
    Subset(dataset_train, list(torch.randperm(len(dataset_train))[:3])),
    batch_size=1,
    shuffle=True,
)


backbone = torch.hub.load(
    "./dinov3",
    "dinov3_vits16",
    source="local",
    weights="https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiMTc0aW56OWd1aW9nZXNoN25pZ2dwNmdsIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU5Njk3NjR9fX1dfQ__&Signature=LTfoFPHH4SWFA7caaZ9aNOiHg5FgGac-fxuQ42Et%7EJdiySSWbkBMdobVjX0PhNlqGiD3VPweg0kl7YO0-ZTywb1VKWfdnRzS-iojQgXjBBegESSNQPaVjVARJo%7E6yDzKU%7Ev-HE-E0P6TkWKx6hvanKT8QpHkzlSqWGBVqu-T13cq%7EWms56RspK%7EsqsEiO9g6j5dUREEERWIQ86OQyLEWWF6WuC%7ECOcuyYyyCXtMnKZQb%7ERptBMEpdtYz8anInDX2mTx37Y97EcYVDay-FiPasUX3T4QKJJ2hCio3e6%7EX6n6Hpl%7EDz0a6dYyOyQ-aF1kOpA7C1MD4xXNYNVl82Qig7A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1831377760851783",
)
backbone = backbone.to(device)
metric_learning_model = DinoMetricLearning(
    backbone,
    dino_out_dim=384,
    cnn_out_dim=128,
    embed_dim=EMBEDDING_SIZE,
    normalize=True,
).to(device)

# metric_learning_model.load_state_dict(torch.load("metric_learning_epoch_5.pth"))

# metric_learning_model = DinoUpsampling(backbone, out_dim=EMBEDDING_SIZE)
# metric_learning_model = metric_learning_model.to(device)
# metric_learning_model.load_state_dict(torch.load("metric_learning_model.pth", map_location=device))

# metric_learning_model = DINOv2_SemanticSegmentation(
#     num_classes=None, image_size=IMAGE_SIZE, embedding_size=EMBEDDING_SIZE
# ).to(device)

# for param in metric_learning_model.feature_extractor.parameters():
#     param.requires_grad = False


# criterion = MagnetLoss(num_classes=NUM_CLASSES, num_clusters=1, margin=0.1)
# criterion = NT_Xent_Loss(temperature=TEMPERATURE)
# class_weights_no_zero = class_weights.clone()
# class_weights_no_zero[0] = 0
# criterion = Proxy_Anchor(num_classes=NUM_CLASSES,
#                          embedding_size=EMBEDDING_SIZE,
#                          class_weights=class_weights_no_zero,
#                          device=device,
#                          use_aux_ce=True,
#                          num_pixels_per_class=400)

# torch.nn.utils.clip_grad_norm_(metric_learning_model.parameters(), 5.0)

# distance = distances.CosineSimilarity()
# reducer = reducers.ThresholdReducer(low=0)
# loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
# mining_func = miners.TripletMarginMiner( margin=0.2, distance=distance, type_of_triplets="semihard" )
# distance = distances.LpDistance(
#     power=2
# )  # Euclidean distance works better with ProxyAnchor
# Alternative: distances.CosineSimilarity() if your embeddings are normalized
distance = distances.CosineSimilarity()

loss_func = losses.ProxyAnchorLoss(
    num_classes=NUM_CLASSES,
    embedding_size=EMBEDDING_SIZE,
    margin=0.1,  # Typical value for ProxyAnchor (higher than triplet margin)
    alpha=30,  # Scaling parameter (controls pull/push strength)
    distance=distance,
).to(device)

model_params = [i for i in metric_learning_model.parameters() if i.requires_grad] + [
    i for i in loss_func.parameters() if i.requires_grad
]


optimizer = torch.optim.Adam(
    model_params,
    lr=LEARNING_RATE,
)

train_metric_learning(
    model=metric_learning_model,
    criterion=loss_func,
    dl_train=dl_train,
    dl_train_small=dl_train_small,
    test_dataset=dataset_test,  # Pass the test dataset for computing test metrics
    # mining_func=mining_func,
    device=device,
    epochs=EPOCHS,
    optimizer=optimizer,
    print_loss=False,
    plot_interval=500,
    metric_interval=4000,
    save_path="metric_learning_model_full_resolution.pth",
)
