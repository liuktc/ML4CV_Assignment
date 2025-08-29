import argparse

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


from pytorch_metric_learning import distances, losses, miners, reducers, testers
from dataset import StreetHazardDataset, PadToMultipleOf16, StreetHazardDatasetTriplet

from model_new import DinoMetricLearning, DinoUpsampling, DinoSegmentation
from train import train_metric_learning
from loss import FixedWeighting, NormalizedWeighting, UncertaintyWeighting


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLAB = False
KAGGLE = False
NUM_CLASSES = 13
SEED = 42
ORIGINAL_IMAGE_SIZE = (720, 1280)

# Seed everything
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument(
    "--lr", type=float, default=1e-4, help="Learning rate for the optimizer"
)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument(
    "--embedding_size",
    type=int,
    default=512,
    help="Size of the embedding vector",
)
parser.add_argument(
    "--pixel_per_class",
    type=int,
    default=50,
    help="Number of pixels to sample per class",
)
parser.add_argument(
    "--scale_factor",
    type=float,
    default=1,
    help="Scale factor for resizing images",
)
parser.add_argument(
    "--preprocess",
    type=str,
    default="resize",
    choices=["crop", "resize"],
    help="Preprocessing method: 'crop' or 'resize'",
)
parser.add_argument(
    "--loss",
    type=str,
    default=None,
    choices=["ProxyAnchor", "TripletMargin", "NTXent"],
    help="Loss function to use",
    required=True,
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="Adam",
    choices=["Adam", "AdamW"],
    help="Optimizer to use",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="Weight decay for AdamW optimizer",
)
parser.add_argument(
    "--normalize_embeddings",
    type=bool,
    default=True,
    help="Whether to normalize embeddings to unit length",
)
parser.add_argument(
    "--model",
    type=str,
    choices=["DinoUpsampling", "DinoMetricLearning"],
    help="Model to use",
    default=None,
    required=True,
)
parser.add_argument(
    "--distance",
    type=str,
    choices=["cos", "l2"],
    help="Distance to use in the loss function",
    default="cos",
)
parser.add_argument(
    "--loss_weighting",
    type=str,
    default="fixed",
    choices=["fixed", "NormalizedWeighting", "UncertaintyWeighting"],
    required=True,
)
parser.add_argument(
    "--lambda_metric",
    type=float,
    default=0.5,
    help="Weight for the metric learning loss",
)
parser.add_argument(
    "--lambda_ce",
    type=float,
    default=1,
    help="Weight for the cross-entropy loss",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="Path to save the best model",
    required=True,
)
parser.add_argument(
    "--plot_interval",
    type=int,
    default=500,
    help="Interval (in batches) to plot embeddings and segmentations",
)

args = parser.parse_args()

DISTANCE = args.distance
MODEL = args.model
EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
EMBEDDING_SIZE = args.embedding_size
PIXEL_PER_CLASS = args.pixel_per_class
SCALE_FACTOR = args.scale_factor
PREPROCESS = args.preprocess
LOSS = args.loss
OPTIMIZER = args.optimizer
WEIGHT_DECAY = args.weight_decay
NORMALIZE_EMBEDDINGS = args.normalize_embeddings
LOSS_WEIGHTING = args.loss_weighting
LAMBDA_METRIC = args.lambda_metric
LAMBDA_CE = args.lambda_ce
SAVE_PATH = args.save_path
PLOT_INTERVAL = args.plot_interval

if not NORMALIZE_EMBEDDINGS and DISTANCE == "cos":
    print(
        "Warning: Using cosine distance without normalizing embeddings. Are you sure?"
    )


image_size = PadToMultipleOf16().convert_dims(
    (ORIGINAL_IMAGE_SIZE[0] * SCALE_FACTOR, ORIGINAL_IMAGE_SIZE[1] * SCALE_FACTOR)
)
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


if PREPROCESS == "resize":
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    target_transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
        ]
    )
elif PREPROCESS == "crop":
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(image_size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    target_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_size),
        ]
    )

# Crea il dataset
dataset_train = StreetHazardDataset(
    annotations_train_file,
    img_dir,
    image_transform=image_transform,
    target_transform=target_transform,
    positive_pairs=False,
)

dataset_val = StreetHazardDataset(
    annotation_val_file,
    img_dir,
    image_transform=image_transform,
    target_transform=target_transform,
    positive_pairs=False,
)

dataset_test = StreetHazardDataset(
    annotation_test_file,
    img_dir_test,
    image_transform=image_transform,
    target_transform=target_transform,
    positive_pairs=False,
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

if MODEL == "DinoUpsampling":
    feature_extractor = DinoUpsampling(
        backbone, out_dim=EMBEDDING_SIZE, normalize=NORMALIZE_EMBEDDINGS
    ).to(device)
    model = DinoSegmentation(
        feature_extractor,
        num_classes=NUM_CLASSES,
    ).to(device)
elif MODEL == "DinoMetricLearning":
    feature_extractor = DinoMetricLearning(
        backbone,
        dino_out_dim=None,
        cnn_out_dim=128,
        out_dim=EMBEDDING_SIZE,
        normalize=NORMALIZE_EMBEDDINGS,
    ).to(device)
    model = DinoSegmentation(
        feature_extractor,
        num_classes=NUM_CLASSES,
    ).to(device)

params_to_train = []

if DISTANCE == "cos":
    distance = distances.CosineSimilarity()
elif DISTANCE == "l2":
    distance = distances.LpDistance(power=2)

if LOSS == "TripletMargin":
    reducer = reducers.ThresholdReducer(low=0)
    metric_learning_loss = losses.TripletMarginLoss(
        margin=0.2, distance=distance, reducer=reducer
    )
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
elif LOSS == "NTXent":
    metric_learning_loss = losses.NTXentLoss(temperature=0.5, distance=distance)
    mining_func = None
elif LOSS == "ProxyAnchor":
    metric_learning_loss = losses.ProxyAnchorLoss(
        num_classes=NUM_CLASSES,
        embedding_size=EMBEDDING_SIZE,
        margin=0.2,
        alpha=30,
        distance=distance,
    ).to(device)
    mining_func = None
    params_to_train += [i for i in metric_learning_loss.parameters() if i.requires_grad]

segmentation_loss = torch.nn.CrossEntropyLoss()


if LOSS_WEIGHTING == "fixed":
    loss_weighting = FixedWeighting(
        lambda_metric=LAMBDA_METRIC,
        lambda_ce=LAMBDA_CE,
    )
elif LOSS_WEIGHTING == "NormalizedWeighting":
    loss_weighting = NormalizedWeighting()
elif LOSS_WEIGHTING == "UncertaintyWeighting":
    loss_weighting = UncertaintyWeighting().to(device)
    params_to_train += [i for i in loss_weighting.parameters() if i.requires_grad]


params_to_train += [i for i in model.parameters() if i.requires_grad]


if OPTIMIZER == "Adam":
    optimizer = torch.optim.Adam(
        params_to_train,
        lr=LEARNING_RATE,
    )
elif OPTIMIZER == "AdamW":
    optimizer = torch.optim.AdamW(
        params_to_train,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

train_metric_learning(
    model=model,
    metric_learning_loss=metric_learning_loss,
    segmentation_loss=segmentation_loss,
    loss_weighting=loss_weighting,
    dl_train=dl_train,
    dl_train_small=dl_train_small,
    test_dataset=dataset_test,  # Pass the test dataset for computing test metrics
    mining_func=mining_func,
    device=device,
    epochs=EPOCHS,
    optimizer=optimizer,
    print_loss=False,
    plot_interval=PLOT_INTERVAL,
    save_path=SAVE_PATH,
    pixel_per_class=PIXEL_PER_CLASS,
)
