import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from plot import plot_semantic_segmentation, semantic_embeddings_plot
from torch.utils.tensorboard import SummaryWriter
from dataset import sample_pixels_per_class
from model_new import OutlierDetector
from metrics import outlier_detection_roc_auc


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path
        self.best_state_dict = None

    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)
                self.best_state_dict = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_semantic_segmentation(
    model: nn.Module,
    criterion: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,  # Add validation DataLoader
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    original_image_size: tuple = (720, 1280),
    plot_interval: int = 1,
    log_dir: str = "./runs",  # TensorBoard log directory
):
    writer = SummaryWriter(log_dir)
    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for i, (images, segmentations, _) in tqdm(enumerate(dl_train)):
            images = images.to(device)
            segmentations = segmentations.to(device)

            logits = model(images)

            # Crop segmentations and logits to the original size
            segmentations = segmentations[
                :, : original_image_size[0], : original_image_size[1]
            ]
            logits = logits[:, :, : original_image_size[0], : original_image_size[1]]

            loss = criterion(logits, segmentations)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            global_step += 1

            if i % plot_interval == 0:
                print(f"Epoch {epoch}/{epochs} - {loss.item()}")
                plot_semantic_segmentation(images[0], logits[0], segmentations[0])

        avg_train_loss = running_train_loss / len(dl_train)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, segmentations, _ in dl_val:
                images = images.to(device)
                segmentations = segmentations.to(device)

                logits = model(images)
                segmentations = segmentations[
                    :, : original_image_size[0], : original_image_size[1]
                ]
                logits = logits[
                    :, :, : original_image_size[0], : original_image_size[1]
                ]

                val_loss = criterion(logits, segmentations)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(dl_val)
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)

        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )

    writer.close()


def train_metric_learning(
    model: nn.Module,
    criterion: nn.Module,
    dl_train: DataLoader,
    dl_train_small: DataLoader,  # Small subset of training data for metrics
    test_dataset: DataLoader,  # Dataset for computing test metrics !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    mining_func=None,
    plot_interval: int = 1,
    metric_interval: int = 100,  # Interval (in batches) to compute test metrics
    log_dir: str = "./runs",  # TensorBoard log directory
    print_loss: bool = False,
    num_classes: int = 13,
    save_path: str = "metric_learning_model.pth",  # Path to save the best model
):
    writer = SummaryWriter(log_dir)
    global_step = 0
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, save_path=save_path)

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for i, (images, segmentations, (selected_pixels, target_matrix)) in tqdm(
            enumerate(dl_train), total=len(dl_train)
        ):
            images = images.to(device)
            segmentations = segmentations.to(device)
            selected_pixels = selected_pixels.to(device)
            target_matrix = target_matrix.to(device)

            embeddings = model(images)
            # Segmentations : BxHxW
            # Embeddings : BxDxHxW
            # Reshape embeddings to (B*H*W) x D
            B, D, H, W = embeddings.shape
            embeddings = embeddings.permute(0, 2, 3, 1).reshape(B * H * W, D)
            # Reshape segmentations to (B*H*W)
            labels = segmentations.view(B * H * W)

            embeddings, labels = sample_pixels_per_class(
                embeddings, labels, num_samples_per_class=50
            )

            if mining_func is None:
                indices_tuple = None
            else:
                indices_tuple = mining_func(embeddings, labels)

            loss = criterion(embeddings, labels, indices_tuple)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            if print_loss:
                print(
                    f"Epoch {epoch}/{epochs} - Step {global_step} - Loss: {loss.item()}"
                )
            global_step += 1

            if i % plot_interval == 0:
                print(f"Epoch {epoch}/{epochs} - {loss.item()}")
                image = semantic_embeddings_plot(
                    model, dl_train, num_points=3000, device=device
                )
                writer.add_image(
                    "Train/Semantic Embeddings",
                    image,
                    global_step=global_step,
                )
                avg_train_loss = running_train_loss / (i + 1)
                early_stopping(avg_train_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    writer.close()
                    return

            if i % metric_interval == 0 and i > 0:
                # Compute test metrics
                detector = OutlierDetector(
                    model,
                    num_classes=num_classes,
                    train_dataloader=dl_train_small,
                    device=device,
                    num_samples_per_class=512,
                    pca_dim=128,
                )

                # Compute the metric over 5 random test images
                scores = []
                test_indices = torch.randperm(len(test_dataset))[:5]
                for idx in tqdm(test_indices, desc="Computing test metrics"):
                    test_image, test_segmentation, _ = test_dataset[idx]
                    outliers_map = detector(test_image.unsqueeze(0).to(device))
                    outliers_gt = (test_segmentation == 13).int()
                    score = outlier_detection_roc_auc(outliers_gt, outliers_map)
                    scores.append(score)
                mean_score = sum(scores) / len(scores)
                writer.add_scalar("Test/Outlier_ROC_AUC", mean_score, global_step)
                print(f"Test Outlier ROC AUC: {mean_score}")

        avg_train_loss = running_train_loss / len(dl_train)
