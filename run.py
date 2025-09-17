import argparse
import json
import os

import torch
from pytorch_metric_learning import distances, losses, miners, reducers
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import wandb
from dataset import PadToMultipleOf16, StreetHazardDataset
from loss import FixedWeighting, NormalizedWeighting, UncertaintyWeighting
from model_new import DinoMetricLearning, DinoSegmentation, DinoUpsampling
from train import train_metric_learning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLAB = False
KAGGLE = False
NUM_CLASSES = 13
SEED = 42
ORIGINAL_IMAGE_SIZE = (720, 1280)

# If this URL is expired, get the new one from
# https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
DINO_DOWNLOAD_URL = r"https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiMWY3cDI4emZrdGZ5NmI2MWZmcjBuemR2IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTc3NzA5NTd9fX1dfQ__&Signature=vKWh9x7KeEwBiHOBalTbkGVwAXE-Nko6w2myuXyqY66Bk9Be%7EVV5qrCuyFxqc%7E4ImtRCg%7EUDNZ2zCWnjI7Gsmic%7E-s-sdmp6roixgPpstDmmMJAXtN-xCpQsiZLCDwhfSLXVcUq-9dvjFoYtkk%7EDyUaI%7EDZorg7gcQl-tuMkB725z5HDdzmrvjJUkO9Vp%7EIG3IONee8%7E7JEE3U3Bq9Ms6NFRu1LEiET%7EW2lzmxVsnCcwRZfLPeq%7Erct1M%7EvZrp0DlHCfdWudxS5FTFzbkW0l9V6FegzHSStd7o-xPv-%7EIRiYpfVRquUvqtdwgKR40T-bLZ38-REPZ%7EceB%7EtpakNFrg__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1866286203927616"
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
parser.add_argument(
    "--kaggle",
    action="store_true",
    help="Whether to run on Kaggle (sets file paths accordingly)",
)
parser.add_argument(
    "--cnn_out_dim",
    type=int,
    default=16,
    help="Output dimension of the CNN in DinoMetricLearning",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="Number of workers for data loading",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    help="Whether to log results to Weights & Biases",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Path to a pretrained model to load",
)
parser.add_argument(
    "--patience",
    type=int,
    default=50,
    help="Patience for early stopping",
)
args = parser.parse_args()

writer = SummaryWriter(
    os.path.join("runs", args.save_path.split("/")[-1].split(".")[0])
)
# Save args to json file
with open(
    os.path.join("runs", f"{args.save_path.split('/')[-1].split('.')[0]}.json"), "w"
) as f:
    json.dump(vars(args), f, indent=4)

writer.add_hparams(
    vars(args),
    {},
)

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
KAGGLE = args.kaggle
CNN_OUT_DIM = args.cnn_out_dim
NUM_WORKERS = args.num_workers
WANDB = args.wandb
PATIENCE = args.patience

if not NORMALIZE_EMBEDDINGS and DISTANCE == "cos":
    print(
        "Warning: Using cosine distance without normalizing embeddings. Are you sure?"
    )

if WANDB:
    wandb.login()
    wandb.init(project="ML4CV_Assignment", entity="luca24ever_unibo", config=vars(args))


image_size = PadToMultipleOf16().convert_dims(
    (ORIGINAL_IMAGE_SIZE[0] * SCALE_FACTOR, ORIGINAL_IMAGE_SIZE[1] * SCALE_FACTOR)
)
# File paths
# annotations_train_file = "/home/federico/.cache/kagglehub/datasets/lucadome/streethazards-train/versions/1/train/train.odgt"
# annotation_val_file = "/home/federico/.cache/kagglehub/datasets/lucadome/streethazards-train/versions/1/train/validation.odgt"
# annotation_test_file = "/home/federico/Downloads/streethazards_test/test/test.odgt"
# img_dir = "/home/federico/.cache/kagglehub/datasets/lucadome/streethazards-train/versions/1/train/"
# img_dir_test = "/home/federico/Downloads/streethazards_test/test/"
if KAGGLE:
    annotations_train_file = "/kaggle/input/streethazards-train/train/train.odgt"
    annotation_val_file = "/kaggle/input/streethazards-train/train/validation.odgt"
    annotation_test_file = "/kaggle/input/streethazards-test/test/test.odgt"
    img_dir = "/kaggle/input/streethazards-train/train/"
    img_dir_test = "/kaggle/input/streethazards-test/test/"
else:
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
dl_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() and NUM_WORKERS > 0 else False,
)
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
    weights=DINO_DOWNLOAD_URL,
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
        cnn_out_dim=CNN_OUT_DIM,
        out_dim=EMBEDDING_SIZE,
        normalize=NORMALIZE_EMBEDDINGS,
    ).to(device)
    model = DinoSegmentation(
        feature_extractor,
        num_classes=NUM_CLASSES,
    ).to(device)

if args.model_path is not None:
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")
else:
    print("Training model from scratch")

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
    writer=writer,
    use_wandb=WANDB,
    use_metric_learning=(LAMBDA_METRIC > 0),
    patience=PATIENCE,
)
